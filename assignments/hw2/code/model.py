import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class PolicyNetwork(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size, activations, out_activation=None, discrete=True,
                 tanh_mean=False, tanh_std=False):
        super(PolicyNetwork, self).__init__()

        assert len(hidden_sizes) > 0, 'No hidden layer exists.'
        assert len(hidden_sizes) == len(activations), 'Num of hidden layers does not match that of activations.'
        self.discrete = discrete

        fc = [nn.Linear(in_size, hidden_sizes[0]), activations[0]]
        for i in range(len(hidden_sizes) - 1):
            fc.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            fc.append(activations[i + 1])

        if self.discrete:
            fc.append(nn.Linear(hidden_sizes[-1], out_size))
        else:
            self.out_size = out_size
            self.mean = nn.Linear(hidden_sizes[-1], out_size)
            self.log_std = nn.Linear(hidden_sizes[-1], out_size)
            self.tanh_mean = tanh_mean
            self.tanh_std = tanh_std
        if out_activation is not None:
            fc.append(out_activation())
        self.fc = nn.Sequential(*fc)

    def forward(self, o):
        if self.discrete:
            logits = self.fc(o)
            probs = F.softmax(logits, dim=1)
            distr = D.Categorical(probs)
            sampled_actions = distr.sample()
            sampled_log_prob = distr.log_prob(sampled_actions)
            return sampled_actions, sampled_log_prob, probs
        else:
            representation = self.fc(o)
            if self.tanh_mean:
                mean = F.tanh(self.mean(representation))
            else:
                mean = self.mean(representation)
            if self.tanh_std:
                log_std = F.tanh(self.log_std(representation))
            else:
                log_std = self.log_std(representation)

            distr = D.Normal(loc=mean, scale=log_std.exp())
            sampled_actions = distr.rsample().detach()
            sampled_log_prob = distr.log_prob(sampled_actions).sum(1).view(-1, 1)
            return sampled_actions, sampled_log_prob, distr

class ValueNetwork(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size, activations, out_activation=None):
        super(ValueNetwork, self).__init__()

        assert len(hidden_sizes) > 0, 'No hidden layer exists.'
        assert len(hidden_sizes) == len(activations), 'Num of hidden layers does not match that of activations.'

        fc = [nn.Linear(in_size, hidden_sizes[0]), activations[0]]
        for i in range(len(hidden_sizes) - 1):
            fc.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            fc.append(activations[i + 1])
        fc.append(nn.Linear(hidden_sizes[-1], out_size))
        if out_activation is not None:
            fc.append(out_activation())
        self.fc = nn.Sequential(*fc)

    def forward(self, o):
        return self.fc(o)
