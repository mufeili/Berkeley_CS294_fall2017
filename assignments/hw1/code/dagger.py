import argparse
import gym
import load_policy
import numpy as np
import pickle
import tensorflow as tf
import tf_util
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from behave_clone import test
from model import FNN
from tf_util import set_tensorboard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/Walker2d-v1_rollouts_20.pkl')
    parser.add_argument('--expert-policy-file', type=str, default='experts/Walker2d-v1.pkl')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--env-name', type=str, default='Walker2d-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max-timesteps', type=int)
    parser.add_argument('--num-rollouts', type=int, default=20)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--num-aggregate', type=int, default=10)
    args = parser.parse_args().__dict__

    args['log_base_dir'] = args['data_file'][5:-4]
    torch.manual_seed(args['random_seed'])

    with open(args['data_file'], 'rb') as f:
        data = pickle.loads(f.read())
    observations = torch.Tensor(data['observations'])
    actions = torch.Tensor(data['actions']).squeeze(1)

    model = FNN(in_size=observations.shape[1],
                hidden1_size=(observations.shape[1]) * 20,
                out_size=actions.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    loss_func = nn.MSELoss()
    _, logger = set_tensorboard(args)

    env = gym.make(args['env_name'])
    env.seed(args['random_seed'])
    max_steps = args['max_timesteps'] or env.spec.timestep_limit

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args['expert_policy_file'])
    print('loaded and built')

    for agger in range(args['num_aggregate'] + 1):
        print('DAgger: ', agger)

        dataset = data_utils.TensorDataset(observations, actions)
        data_loader = data_utils.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)
        model.train()
        for epoch in range(args['num_epochs']):
            print('Starting epoch: {}'.format(epoch + 1))
            for batch_idx, (obs_batch, act_batch) in enumerate(data_loader):
                loss = loss_func(model(obs_batch), act_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Setup tensorboard
                logger.log_value('mse_loss', loss.detach().item(),
                                 (agger * args['num_epochs'] + epoch) * int(len(dataset)/args['batch_size'])
                                 + batch_idx + 1)

        if agger < args['num_aggregate']:
            model.eval()
            returns = []
            new_obs_ = []
            for rollout in range(args['num_rollouts']):
                print('rollout: ', rollout)
                obs_ = env.reset()
                done = False
                totalr = 0.
                steps = 0

                while not done:
                    new_obs_.append(obs_)
                    obs = torch.Tensor([obs_])
                    action = model(obs).detach().numpy()
                    obs_, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1

                    if args['render']:
                        env.render()
                    if steps % 100 == 0:
                        print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            return_mean = np.mean(returns)
            return_std = np.std(returns)
            print('mean return', return_mean)
            print('std of return', return_std)
            logger.log_value('return mean', float(return_mean), agger)
            logger.log_value('return std', float(return_std), agger)

            new_obs_ = np.stack(new_obs_)
            with tf.Session():
                tf_util.initialize()
                new_acts_ = policy_fn(new_obs_)
            new_obs = torch.Tensor(new_obs_)
            new_acts = torch.Tensor(new_acts_)
            observations = torch.cat((observations, new_obs), 0)
            actions = torch.cat((actions, new_acts), 0)

    if args['test']:
        model.eval()
        test(model, args, logger)

if __name__ == '__main__':
    main()
