#!/usr/bin/env bash
# Download the 4 source match videos (public GitHub repositories) into videos/
set -e
mkdir -p videos tmp_src && cd tmp_src
get() {  # repo  path-in-repo  output-name
  d=$(basename "$1")
  [ -d "$d" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --filter=blob:none --no-checkout "https://github.com/$1" "$d"
  git -C "$d" checkout HEAD -- "$2" && cp "$d/$2" "../videos/$3"
}
get Dayallenr/Football-Tracking-and-Analysis-System input_videos/08fd33_4.mp4 08fd33_4.mp4
get Hazrat-Ali9/Real-Time-Football-Player-Team-Tracking foatball/data/121364_0.mp4 121364_0.mp4
get ConnorSiXiong/Kaggle_DFL_BundesligaDataShootout data/cut_1.mp4 cut_1.mp4
get Mostafa-Nafie/Football-Object-Detection test_videos/CityUtdR.mp4 CityUtdR.mp4
cd .. && rm -rf tmp_src && ls -la videos
