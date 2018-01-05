#!/bin/bash

# Typical usage, `watch ./monitoring.sh`

echo -e "\nGPU mem:\n"
cuda-smi

echo -e "\nLoss:\n"
tail results/loss.csv

echo -e "\nTrain accuracy:\n"
tail results/train_acc.csv

echo -e "\nValidation accuracy:\n"
tail results/val_acc.csv

echo -e "\nTest accuracy:\n"
tail results/test_acc.csv
