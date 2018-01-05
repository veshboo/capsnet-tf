#!/bin/bash

# Typica usage: ./train.sh --epoch={number-of-epoch}

python main.py --save_freq=1 --batch_size=16 $@
