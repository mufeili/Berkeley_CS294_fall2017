#!/usr/bin/env bash
python run_expert.py experts/Walker2d-v1.pkl Walker2d-v1 --num_rollouts 20
python dagger.py --data-file data/Walker2d-v1_rollouts_20.pkl --expert-policy-file experts/Walker2d-v1.pkl --batch-size\
        64 --lr 1e-3 --num-epochs 10 --test --env-name Walker2d-v1 --random-seed 0 --num-aggregate 10