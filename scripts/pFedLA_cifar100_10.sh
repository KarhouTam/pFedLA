#!/bin/bash

python ../src/data/run.py \
 --dataset cifar100 \
 --client_num_in_total 10 \
 --classes 12;

sleep 2;

python ../src/server/pFedLA.py \
 --client_num_per_round 10 \
 --dataset cifar100 \
 --global_epochs 100 \
 --local_epochs 10 \
 --log 1 \
 --gpu 1 \
 --save_period 20 ;