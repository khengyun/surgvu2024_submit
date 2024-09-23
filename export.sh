#!/usr/bin/env bash

./build.sh

docker save surgtoolloc_det:latest | gzip -c > surgtoolloc_det.tar.gz
