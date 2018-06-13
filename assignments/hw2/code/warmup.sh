#!/usr/bin/env bash
python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -dna -l 1 -s 32
    --exp_name sb_no_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg -dna -l 1 -s 32
    --exp_name sb_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg -l 1 -s 32
    --exp_name sb_rtg_na
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -dna -l 1 -s 32
    --exp_name lb_no_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg -dna -l 1 -s 32
    --exp_name lb_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg -l 1 -s 32
    --exp_name lb_rtg_na
