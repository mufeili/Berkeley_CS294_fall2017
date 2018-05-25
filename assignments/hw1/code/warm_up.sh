#!/usr/bin/env bash
python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 20
python behave_clone.py --data-file data/Hopper-v1_rollouts_20.pkl --batch-size 64 --lr 1e-3 --num-epochs 10 \
 --test --render