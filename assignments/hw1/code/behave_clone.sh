#!/usr/bin/env bash
python run_expert.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 20
python behave_clone.py --data-file data/Ant-v1_rollouts_20.pkl --batch-size 64 --lr 1e-3 --num-epochs 10 \
 --test --env-name Ant-v1 --render --random-seed 0