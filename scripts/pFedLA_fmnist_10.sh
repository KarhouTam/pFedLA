#!/bin/bash

python ../src/data/run.py \
 --dataset fmnist \
 --client_num_in_total 10 \
 --classes 4;

sleep 2;

python ../src/server/pFedLA.py \
 --client_num_per_round 10 \
 --dataset fmnist \
 --global_epochs 100 \
 --local_epochs 10 \
 --log 1 \
 --gpu 1 \
 --save_period 20 ;