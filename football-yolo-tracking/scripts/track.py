"""
Soccer object detection + tracking + team assignment with a fine-tuned YOLOv8 model.

Pass 1 (detect + track)
- player / referee: ByteTrack (model.track(..., persist=True, agnostic_nms=True)); the class shown for a
  track is a confidence-weighted vote over its history, so one person does not flicker
  between player and referee; tracks seen in < 5 frames are dropped
- ball: tiny, so detected separately at a higher resolution; one ball per frame, kept
  only if consistent with the previous ball position
- for every player box: jersey colour = median Lab colour of the upper torso with
  grass-coloured pixels removed

Team assignment
- K-means (k=2) on the chroma (a, b) of each player track's median jersey colour
  (lightness is left out: shade and sunlight change it, team colours do not)
- every player track joins the nearest team centre

Pass 2 (draw)
- ellipse + ID in the team colour, referee in red, ball as a yellow triangle,
  trajectories, HUD with per-team counts
- trajectories saved to CSV: frame, track_id, class, team, cx, cy, w, h, conf

Usage:
    python scripts/track.py --weights weights/best.pt --source input/match_5s.mp4 \
        --out results/tracking_5s.mp4 --csv results/trajectories.csv
"""
import argparse
import csv
from collections import defaultdict, deque

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

REF_COLOR = (60, 60, 230)    # BGR
BALL_COLOR = (0, 215, 255)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--source", default="input/match_5s.mp4")
    p.add_argument("--out", default="results/tracking_5s.mp4")
    p.add_argument("--csv", default="results/trajectories.csv")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--trail", type=int, default=40, help="trajectory length in frames")
    p.add_argument("--ball-imgsz", type=int, default=1280, help="ball is tiny: detect it at a higher resolution")
    p.add_argument("--ball-conf", type=float, default=0.15)
    p.add_argument("--ball-max-jump", type=float, default=60, help="max px the ball may move per frame")
    p.add_argument("--min-track-len", type=int, default=5, help="drop tracks seen in fewer frames (false positives)")
    p.add_argument("--ball-max-dist", type=float, default=300, help="max px from the last ball position, however long the gap")
    return p.parse_args()


def grass_lab(frame):
    """Median Lab colour of the pitch (middle band of the image)."""
    h = frame.shape[0]
    band = frame[int(h * .3):int(h * .85)]
    return np.median(cv2.cvtColor(band, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float), 0)


def jersey_lab(frame, box, grass):
    """Median Lab colour of the upper torso, ignoring grass-coloured pixels."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    torso = frame[y1 + int(h * .15):y1 + int(h * .45), x1 + int(w * .2):max(x1 + int(w * .2) + 1, x2 - int(w * .2))]
    if torso.size == 0:
        return None
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    keep = lab[np.linalg.norm(lab - grass, axis=1) > 25]
    return np.median(keep if len(keep) > 3 else lab, 0)


def lab_to_bgr(lab):
    px = np.uint8([[np.clip(lab, 0, 255)]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])


def detect_and_track(a, model):
    """Pass 1: returns per-frame detections and per-track jersey colours."""
    cap = cv2.VideoCapture(a.source)
    frames_det, colors = [], defaultdict(list)
    votes = defaultdict(lambda: defaultdict(float))
    last_ball, fi = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = []
        # agnostic_nms: one box per person (otherwise the same person can get both a
        # "player" and a "referee" box, because NMS is run per class by default)
        res = model.track(frame, persist=True, imgsz=a.imgsz, conf=a.conf, classes=[1, 2],
                          agnostic_nms=True, tracker=a.tracker, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            grass = grass_lab(frame)
            for box, c, s, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                                      res.boxes.conf.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int)):
                votes[tid][c] += float(s)
                c = max(votes[tid], key=votes[tid].get)        # track-level class vote
                if c == 1:
                    col = jersey_lab(frame, box, grass)
                    if col is not None:
                        colors[tid].append(col)
                dets.append([box, c, float(s), int(tid)])

        # ball: single object, high-res detection + position consistency
        bres = model.predict(frame, imgsz=a.ball_imgsz, conf=a.ball_conf, classes=[0], verbose=False)[0]
        cand = []
        for box, s in zip(bres.boxes.xyxy.cpu().numpy(), bres.boxes.conf.cpu().numpy()):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if last_ball is not None:
                gap = fi - last_ball[2]
                if np.hypot(cx - last_ball[0], cy - last_ball[1]) > min(a.ball_max_jump * gap, a.ball_max_dist):
                    continue
            cand.append((float(s), box, cx, cy))
        if cand:
            s, box, cx, cy = max(cand, key=lambda t: t[0])
            dets.append([box, 0, s, 0])
            last_ball = (cx, cy, fi)
        if last_ball is not None and fi - last_ball[2] > 25:
            last_ball = None  # lost for >1 s: allow re-acquiring anywhere

        # the final class of a track is only known later -> store the vote table reference
        frames_det.append(dets)
        fi += 1
    cap.release()
    final_cls = {tid: max(v, key=v.get) for tid, v in votes.items()}
    # drop very short tracks: ad boards, pitch lines, one-off false positives
    length = defaultdict(int)
    for dets in frames_det:
        for d in dets:
            if d[1] != 0:
                length[d[3]] += 1
    frames_det = [[d for d in dets if d[1] == 0 or length[d[3]] >= a.min_track_len] for dets in frames_det]
    colors = {t: v for t, v in colors.items() if length[t] >= a.min_track_len}
    return frames_det, colors, final_cls


def assign_teams(colors, final_cls):
    """K-means on jersey chroma (a, b) of player tracks -> {track_id: 0/1}, team colours (BGR)."""
    tids = [t for t, v in colors.items() if final_cls.get(t) == 1 and len(v) > 0]
    med = np.array([np.median(colors[t], 0) for t in tids])
    weight = np.array([len(colors[t]) for t in tids], float)
    km = KMeans(2, n_init=10, random_state=0).fit(med[:, 1:], sample_weight=weight)
    team = {t: int(l) for t, l in zip(tids, km.predict(med[:, 1:]))}
    # team colour for drawing = weighted mean jersey colour of its members
    team_bgr = []
    for k in range(2):
        m = [i for i, t in enumerate(tids) if team[t] == k]
        lab = np.average(med[m], axis=0, weights=weight[m])
        team_bgr.append(lab_to_bgr(lab))
    # name teams A/B so that A is the lighter kit (stable across runs)
    if sum(team_bgr[0]) < sum(team_bgr[1]):
        team = {t: 1 - k for t, k in team.items()}
        team_bgr = team_bgr[::-1]
    return team, team_bgr


def draw_label(frame, text, anchor, fill):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = anchor
    cv2.rectangle(frame, (x - tw // 2 - 3, y + 9), (x + tw // 2 + 3, y + 13 + th), fill, -1)
    txt = (0, 0, 0) if sum(fill) > 300 else (255, 255, 255)
    cv2.putText(frame, text, (x - tw // 2, y + 11 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt, 1, cv2.LINE_AA)


def main():
    a = parse_args()
    model = YOLO(a.weights)
    frames_det, colors, final_cls = detect_and_track(a, model)
    team, team_bgr = assign_teams(colors, final_cls)
    team_name = ["A", "B"]

    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trails = defaultdict(lambda: deque(maxlen=a.trail))
    ball_trail = deque(maxlen=a.trail)
    rows = []

    for fi, dets in enumerate(frames_det):
        ok, frame = cap.read()
        if not ok:
            break
        counts = defaultdict(int)
        for box, c, s, tid in dets:
            x1, y1, x2, y2 = [int(v) for v in box]
            if c == 0:
                counts["ball"] += 1
                cx = (x1 + x2) // 2
                ball_trail.append((cx, (y1 + y2) // 2, fi))
                pts = np.array([[cx, y1 - 4], [cx - 9, y1 - 20], [cx + 9, y1 - 20]])
                cv2.fillPoly(frame, [pts], BALL_COLOR)
                cv2.polylines(frame, [pts], True, (0, 0, 0), 1)
                rows.append([fi, 0, "ball", "", (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, round(s, 3)])
                continue
            c = final_cls.get(tid, c)  # final vote of the whole track
            if c == 1 and tid in team:
                k = team[tid]
                color, label, tname = team_bgr[k], f"{team_name[k]}{tid}", team_name[k]
                counts[f"team{tname}"] += 1
            elif c == 2:
                color, label, tname = REF_COLOR, f"R{tid}", ""
                counts["referee"] += 1
            else:
                color, label, tname = (200, 200, 200), f"P{tid}", ""
                counts["player?"] += 1
            foot = ((x1 + x2) // 2, y2)
            trails[tid].append(foot)
            cv2.ellipse(frame, foot, (max(10, (x2 - x1) // 2 + 4), 7), 0, -45, 235, (0, 0, 0), 4)
            cv2.ellipse(frame, foot, (max(10, (x2 - x1) // 2 + 4), 7), 0, -45, 235, color, 2)
            draw_label(frame, label, foot, color)
            rows.append([fi, tid, "player" if c == 1 else "referee", tname,
                         (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, round(s, 3)])

        for tid, pts in trails.items():
            if len(pts) > 1:
                cv2.polylines(frame, [np.array(pts, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
        bt = [p for p in ball_trail if fi - p[2] <= a.trail]
        for p0, p1 in zip(bt, bt[1:]):
            if p1[2] - p0[2] <= 8:  # no line across long gaps
                cv2.line(frame, p0[:2], p1[:2], BALL_COLOR, 2, cv2.LINE_AA)

        # HUD with a colour chip per team
        cv2.rectangle(frame, (20, 20), (760, 62), (0, 0, 0), -1)
        x = 32
        for k in range(2):
            cv2.rectangle(frame, (x, 30), (x + 22, 52), team_bgr[k], -1)
            txt = f"Team {team_name[k]} {counts['team' + team_name[k]]}"
            cv2.putText(frame, txt, (x + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            x += 210
        cv2.rectangle(frame, (x, 30), (x + 22, 52), REF_COLOR, -1)
        cv2.putText(frame, f"Referee {counts['referee']}", (x + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        x += 190
        cv2.putText(frame, f"Ball {counts['ball']}", (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BALL_COLOR, 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    with open(a.csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "track_id", "class", "team", "cx", "cy", "w", "h", "conf"])
        wr.writerows([[r[0], r[1], r[2], r[3], round(r[4], 1), round(r[5], 1), r[6], r[7], r[8]] for r in rows])
    sizes = {team_name[k]: sum(1 for v in team.values() if v == k) for k in range(2)}
    print(f"done: {len(frames_det)} frames -> {a.out}, {len(rows)} rows -> {a.csv}, team tracks {sizes}, team colours (BGR) {team_bgr}")


if __name__ == "__main__":
    main()
