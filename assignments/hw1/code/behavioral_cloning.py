import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from model import FNN
from tensorboard_util import Logger
from torch.autograd import Variable


def main():

    def test(policy, setting):
        import gym

        env = gym.make(setting.envname)
        max_steps = setting.max_timesteps or env.spec.timestep_limit
        returns = []

        for rollout in range(setting.num_rollouts):
            print('rollout', rollout)
            obs_ = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                obs = Variable(torch.Tensor([obs_]), requires_grad=False)
                action = policy(obs).data.squeeze(0).numpy()
                obs_, r, done, _ = env.step(action)
                totalr += r
                steps += 1

                if setting.render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/Hopper-v1_num_rollouts_20.pkl')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--envname', type=str, default='Hopper-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    data = pickle.loads(open(args.data_file, 'rb').read())
    observations = torch.stack([torch.Tensor(o) for o in data['observations']])
    actions = torch.stack([torch.Tensor(a) for a in data['actions']]).squeeze(1)
    data_set = data_utils.TensorDataset(observations, actions)
    data_loader = data_utils.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

    model = FNN(input_size=observations.shape[1], hidden1_size=(observations.shape[1] - 1) * 20,
                output_size=actions.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.MSELoss()
    logger = Logger('/'.join(['tensorboard_log', args.data_file[5:-4]]))

    model.train()
    for epoch in range(args.num_epochs):
        print('Starting epoch: {}'.format(epoch + 1))
        for batch_idx, (obs_batch_, act_batch_) in enumerate(data_loader):
            obs_batch = Variable(obs_batch_, requires_grad=False)
            act_batch = Variable(act_batch_, requires_grad=False)
            act_pred = model(obs_batch)

            loss = loss_function(act_pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.scalar_summary('mse_loss', loss.data[0], epoch * int(len(data_set)/100) + batch_idx + 1)

    # Now you can start tensorboard and check the learning curve.

    if args.test:
        test(model, args)


if __name__ == '__main__':
    main()
