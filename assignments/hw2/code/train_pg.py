import numpy as np
import torch
import torch.nn as nn
import gym
import logz
import os
import time
import inspect
from model import PolicyNetwork, ValueNetwork
from multiprocessing import Process
from torch.optim import Adam

#============================================================================================#
# Utilities
#============================================================================================#

def policy_gradient_loss(log_prob, adv, num_path):
    return - (log_prob.view(-1, 1) * adv).sum() / num_path

def build_network(
        input_size,
        output_size,
        discrete=True,
        network='policy',
        n_layers=1,
        size=32,
        activation=nn.Tanh,
        output_activation=None,
        tanh_mean=False,
        tanh_std=False
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units. 
    # 
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    #========================================================================================#

    # YOUR_CODE_HERE
    if network == 'policy':
        return PolicyNetwork(input_size, [size for _ in range(n_layers)], output_size,
                             [activation() for _ in range(n_layers)], output_activation, discrete, tanh_mean, tanh_std)
    elif network == 'value':
        return ValueNetwork(input_size, [size for _ in range(n_layers)], 1,
                            [activation() for _ in range(n_layers)], output_activation)

def pathlength(path):
    return len(path["reward"])



#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=2,
             size=64,
             tanh_mean=False,
             tanh_std=False,
             int_activation='relu'
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.signature(train_PG).parameters.keys()
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    # 
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over 
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use torch.distributions.Normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken, 
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the 
    #      policy network output ops.
    #   
    #========================================================================================#

    if int_activation == 'relu':
        activation = nn.ReLU
    elif int_activation == 'tanh':
        activation = nn.Tanh

    if discrete:
        policy = build_network(input_size=ob_dim, output_size=ac_dim, discrete=discrete, network='policy',
                               n_layers=n_layers, size=size, activation=activation)
    else:
        policy = build_network(input_size=ob_dim, output_size=ac_dim, discrete=discrete, network='policy',
                               n_layers=n_layers, size=size, tanh_mean=tanh_mean, tanh_std=tanh_std,
                               activation=activation)

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    policy_loss = policy_gradient_loss # Loss function that we'll differentiate to get the policy gradient.
    policy_optimizer = Adam(policy.parameters(), lr=learning_rate)

    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = build_network(input_size=ob_dim, output_size=1, network='value', n_layers=n_layers,
                                            size=size, activation=activation)
        baseline_loss = nn.MSELoss()
        baseline_optimizer = Adam(baseline_prediction.parameters(), lr=learning_rate)

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob_ = env.reset()
            obs, acs, rewards, log_probs = [], [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                ob = torch.from_numpy(ob_).float().unsqueeze(0)
                obs.append(ob)
                ac_, log_prob, _ = policy(ob)
                acs.append(ac_)
                log_probs.append(log_prob)
                if discrete:
                    ac = int(ac_)
                else:
                    ac = ac_.squeeze(0).numpy()
                ob_, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : torch.cat(obs, 0),
                    "reward" : torch.Tensor(rewards),
                    "action" : torch.cat(acs, 0),
                    "log_prob" : torch.cat(log_probs, 0)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = torch.cat([path["observation"] for path in paths], 0)
        ac_na = torch.cat([path["action"] for path in paths], 0)

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above). 
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where 
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t. 
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG 
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        #       entire trajectory (regardless of which time step the Q-value should be for). 
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG 
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above. 
        #
        #====================================================================================#

        # YOUR_CODE_HERE
        q_n_ = []
        for path in paths:
            rewards = path['reward']
            num_steps = pathlength(path)
            if reward_to_go:
                q_n_.append(torch.cat([(torch.pow(gamma, torch.arange(num_steps - t)) * rewards[t:]).sum().view(-1, 1)
                                       for t in range(num_steps)]))
            else:
                q_n_.append((torch.pow(gamma, torch.arange(num_steps)) * rewards).sum() * torch.ones(num_steps, 1))
        q_n = torch.cat(q_n_, 0)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)
            b_n = baseline_prediction(ob_no)
            q_n_std = q_n.std()
            q_n_mean = q_n.mean()
            b_n_scaled = b_n * q_n_std + q_n_mean
            adv_n = (q_n - b_n_scaled).detach()
        else:
            adv_n = q_n

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
            # YOUR_CODE_HERE
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + np.finfo(np.float32).eps.item())


        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            target = (q_n - q_n_mean) / (q_n_std + np.finfo(np.float32).eps.item())
            baseline_optimizer.zero_grad()
            b_loss = baseline_loss(b_n, target)
            b_loss.backward()
            baseline_optimizer.step()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE
        log_probs = torch.cat([path["log_prob"] for path in paths], 0)
        policy_optimizer.zero_grad()
        loss = policy_loss(log_probs, adv_n, len(paths))
        loss.backward()
        policy_optimizer.step()

        """
        # Code for checking loss and debugging
        
        if discrete:
            _, _, log_probs_new_ = policy(ob_no)
            log_probs_new = log_probs_new_.gather(1, ac_na.view(-1, 1))
        else:
            _, _, distr = policy(ob_no)
            log_probs_new = distr.log_prob(ac_na.view(-1, 1)).sum(1).view(-1, 1)
        loss_new = policy_loss(log_probs_new, adv_n, len(paths))
        """

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        if nn_baseline:
            logz.save_checkpoint({
                'policy_state_dict': policy.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'baseline_state_dict': baseline_prediction.state_dict(),
                'baseline_optimizer': baseline_optimizer.state_dict()
            })
        else:
            logz.save_checkpoint({
                'policy_state_dict': policy.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict()
            })

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--tanh_mean', '-tm', action='store_true')
    parser.add_argument('--tanh_std', '-ts', action='store_true')
    parser.add_argument('--internal_activation', '-ia', type=str, default='tanh')
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                tanh_mean=args.tanh_mean,
                tanh_std=args.tanh_std,
                int_activation = args.internal_activation
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()
