import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from model import FNN
from tensorboard_util import Logger
from torch.autograd import Variable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='data/Hopper-v1_num_rollouts_20_max_steps_1000.pkl')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-epochs', type=int, default=10)
    args = parser.parse_args()

    data = pickle.loads(open(args.data_file, 'rb').read())
    observations = torch.stack([torch.Tensor(o) for o in data['observations']])
    actions = torch.stack([torch.Tensor(a) for a in data['actions']]).squeeze(1)
    data_set = data_utils.TensorDataset(observations, actions)
    data_loader = data_utils.DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

    model = FNN(input_size=observations.shape[1], output_size=actions.shape[1])
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


if __name__ == '__main__':
    main()
