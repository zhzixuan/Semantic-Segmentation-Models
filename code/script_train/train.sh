#!/usr/bin/env sh

/home/user/workspace/py-RFCN-priv/caffe-priv/build/tools/caffe train --gpu=all \
     --solver=./solver.prototxt \
     --weights=./se-resnet50-hik-merge.caffemodel


