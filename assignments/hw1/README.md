# README

---
# Dependencies

- Python 3.5.5
- [TensorFlow 1.1.0](https://www.tensorflow.org/)
- [PyTorch 0.4.0](http://pytorch.org/)
- [MuJuCo 1.31](http://www.mujoco.org/)
- [OpenAI Gym 0.9.3](https://github.com/openai/gym) -- Note you also need to install [mujoco-py 0.5.7](https://github.com/openai/mujoco-py/tree/master) for Gym.
- [tensorboard_logger 0.1.0](https://github.com/TeamHG-Memex/tensorboard_logger)

# Warmup

- Run `sh warm_up.sh`
- Launch the tensorboard to view the learning curve.

# Behavioral Cloning

## Part I

- Run `sh behave_clone.sh`
- Launch the tensorboard to view the learning curve.

To obtain the results for the two environments, one needs to modify the corresponding arguments in behave_clone.sh.

## Part II

This just requires changing the random seeds in behave_clone.sh.

# DAgger

- Run `sh dagger.sh`

`error_bar.py` was used for plotting error bar plots.




