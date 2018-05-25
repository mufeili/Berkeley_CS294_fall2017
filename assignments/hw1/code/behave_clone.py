import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from model import FNN
from tf_util import set_tensorboard


def test(policy, setting, logger):
    import gym

    env = gym.make(setting['env_name'])
    env.seed(setting['random_seed'])
    max_steps = setting['max_timesteps'] or env.spec.timestep_limit
    returns = []

    for rollout in range(setting['num_rollouts']):
        print('rollout: ', rollout)
        obs_ = env.reset()
        done = False
        totalr = 0.
        steps = 0

        while not done:
            obs = torch.Tensor([obs_])
            action = policy(obs).detach().numpy()
            obs_, r, done, _ = env.step(action)
            totalr += r
            steps += 1

            if setting['render']:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        logger.log_value('return', totalr, rollout + 1)
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/Hopper-v1_rollouts_20.pkl')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--env-name', type=str, default='Hopper-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max-timesteps', type=int)
    parser.add_argument('--num-rollouts', type=int, default=20)
    parser.add_argument('--random-seed', type=int, default=0)
    args = parser.parse_args().__dict__

    args['log_base_dir'] = args['data_file'][5:-4]
    torch.manual_seed(args['random_seed'])

    with open(args['data_file'], 'rb') as f:
        data = pickle.loads(f.read())
    observations = torch.Tensor(data['observations'])
    actions = torch.Tensor(data['actions']).squeeze(1)
    dataset = data_utils.TensorDataset(observations, actions)
    data_loader = data_utils.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

    model = FNN(in_size=observations.shape[1],
                hidden1_size=(observations.shape[1]) * 20,
                out_size=actions.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    loss_func = nn.MSELoss()
    _, logger = set_tensorboard(args)

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
                             epoch * int(len(dataset)/args['batch_size']) + batch_idx + 1)

    if args['test']:
        model.eval()
        test(model, args, logger)

if __name__ == '__main__':
    main()
