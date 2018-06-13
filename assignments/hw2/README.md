# README

---
# Dependencies

- Python 3.5.5
- [TensorFlow 1.1.0](https://www.tensorflow.org/)
- [PyTorch 0.4.0](http://pytorch.org/)
- [OpenAI Gym 0.9.3](https://github.com/openai/gym) 

For the visualization required, just follow the instructions in homework requirement and the comments in `plot.py`.

# 4 Implement Policy Gradient

## 4.2.1

- Run `sh warmup.sh`

## 4.2.2

- Run `python train_pg.py InvertedPendulum-v1 -n 100 -b 2000 -l 1 -s 32 -e 1 -rtg 
    -lr 5e-3 -ts -tm --exp_name b2000_l1_s32`

# 5  Implement Neural Network Baselines

- Run ```console
python train_pg.py InvertedPendulum-v1 -n 100 -b 2000 -l 1 -s 32 -e 10 -rtg -lr 5e-3 -ts -tm --exp_name b2000
python train_pg.py InvertedPendulum-v1 -n 100 -b 2000 -l 1 -s 32 -e 10 -rtg -lr 5e-3 -ts -tm -bl  --exp_name b2000_baseline```

# 6 HalfCheetah

- Run `python train_pg.py HalfCheetah-v1 -ep 150 --discount 0.9 -n 100 -b 50000 -l 1 -s 32 -e 1 -rtg -lr 0.025 -ts -tm -bl -ia relu --exp_name b50000_l1_s32_e1_rtg_ts_tm_bl_relu`
