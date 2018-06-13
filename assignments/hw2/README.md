# README

---
# Dependencies

- Python 3.5.5
- [PyTorch 0.4.0](http://pytorch.org/)
- [MuJuCo 1.31](http://www.mujoco.org/)
- [OpenAI Gym 0.9.3](https://github.com/openai/gym) -- Note you also need to install [mujoco-py 0.5](https://github.com/openai/mujoco-py/tree/0.5) for Gym.

For the visualization required, just follow the instructions in homework requirement and the comments in `plot.py`.

# 4 Implement Policy Gradient

## 4.2.1

- Run `sh warmup.sh`

## 4.2.2

- Run `python train_pg.py InvertedPendulum-v1 -n 100 -b 2000 -l 1 -s 32 -e 1 -rtg 
    -lr 5e-3 -ts -tm --exp_name b2000_l1_s32`

# 5  Implement Neural Network Baselines

- Run `
python train_pg.py InvertedPendulum-v1 -n 100 -b 2000 -l 1 -s 32 -e 10 -rtg -lr 5e-3 -ts -tm --exp_name b2000` and `
python train_pg.py InvertedPendulum-v1 -n 100 -b 2000 -l 1 -s 32 -e 10 -rtg -lr 5e-3 -ts -tm -bl  --exp_name b2000_baseline`

# 6 HalfCheetah

- Run `python train_pg.py HalfCheetah-v1 -ep 150 --discount 0.9 -n 100 -b 50000 -l 1 -s 32 -e 1 -rtg -lr 0.025 -ts -tm -bl -ia relu --exp_name b50000_l1_s32_e1_rtg_ts_tm_bl_relu`
