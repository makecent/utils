#!/usr/bin/env bash

VIDEO_DIR=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/dense_feature_extraction.py $VIDEO_DIR \
    --launcher pytorch "${@:3}"