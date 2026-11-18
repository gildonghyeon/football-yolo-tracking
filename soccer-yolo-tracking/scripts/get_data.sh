#!/usr/bin/env bash
# Download the 3-class soccer dataset (ball / player / referee, YOLO format)
set -e
git clone --depth 1 https://github.com/skaczylo/football-image-dataset data/football-image-dataset
