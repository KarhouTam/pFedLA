#!/bin/bash

python ../src/data/run.py \
 --dataset cifar10 \
 --client_num_in_total 100 \
 --classes 4;

sleep 2;

python ../src/server/pFedLA.py \
 --client_num_per_round 10 \
 --dataset cifar10 \
 --global_epochs 250 \
 --local_epochs 20 \
 --log 1 \
 --gpu 1 \
 --save_period 20 ;