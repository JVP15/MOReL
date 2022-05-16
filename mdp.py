import os

import time

import argparse

import torch
import torch.nn as nn
import numpy as np
import pickle
import reward_functions
from dynamics_model import DynamicsModel
from usad import USAD


class MDP(object):
    def __init__(self,
                 dataset,
                 env_name,
                 negative_reward = 100,
                 std = 0.01,
                 num_epochs = 300,
                 batch_size = 256,
                 learning_rate = 5e-4,
                 device='cpu',
                 model_path=None,
                 usad_folder=None):
        """Models a pessimistic MDP for the MOReL algorithm.
        :param dataset: The dataset to use for training. It is expected to be a list of trajectories where each
        trajectory is a dictionary of the form {'observations': [], 'actions': [], 'rewards': []}.
        :param env: The environment that the MDP is trained on. This is used solely for the reward function,
        although if you aren't going to call the reward function, you can leave it as None

        :param model_path: The path to a saved dynamics model.

        Interface for this class is (mostly) taken from the WorldModel class from MBRL:
        https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L7"""

        # set the environment for the reward function. We're expecting a GymEnv from MJRL, but we only need the gym.Env object that it wraps
        self.env_name = env_name

        self.device = device

        self.std = std

        self.action_size = 0
        self.state_size = 0

        # this is from the WorldModel class. It tells the planning algorithm whether the to use the MDP's reward
        #  function or a different one
        self.learn_reward = True
        self.min_reward = np.inf

        self._init_statistics(dataset)
        self.negative_reward = self.min_reward - negative_reward

        # for simplicity's sake, we'll just set the absorbing state to be all 0s. I'm not sure if this is how they
        #   actually did it in the MOReL paper, but it is what I am going with for now.
        self.absorbing_state = torch.zeros(self.state_size, device=self.device)

        self.dynamics_model = DynamicsModel(dataset, std, device)
        if model_path:
            print(f'MDP: Loading dynamics model from {model_path}')
            self.dynamics_model.load_state_dict(torch.load(model_path))
        else:
            self.dynamics_model.fit(num_epochs, batch_size, learning_rate)

        self.loss_func = nn.MSELoss()

        # if the USAD folder is none, then we'll train the USAD using the dataset, otherwise it will load
        #   the pretrained dynamics models from the folder
        self.usad = USAD(dataset, self.state_size, self.action_size,
                         num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
                         device=device, usad_folder=usad_folder)

    def _init_statistics(self, dataset):
        for trajectory in dataset:
            if np.min(trajectory['rewards']) < self.min_reward:
                self.min_reward = np.min(trajectory['rewards'])

        self.action_size = len(dataset[0]['actions'][0])
        self.state_size = len(dataset[0]['observations'][0])

    def to(self, device):
        self.dynamics_model.to(device)
        self.absorbing_state.to(device)

    def is_cuda(self):
        return self.dynamics_model.device.startswith('cuda')

    def forward(self, s, a):
        #known_state_action_pairs = self.usad(s, a)

        # I had to create different functions for batches of states and actions in the forward pass
        # if hasattr(known_state_action_pairs, '__len__'):
        #     return self._forward_batch(s, a, known_state_action_pairs)
        # else:
        #     return self._forward_single(s, a, known_state_action_pairs)

        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float().to(self.device)
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float().to(self.device)

        return self.dynamics_model.predict(s, a)

    def _forward_batch(self, s, a, known_state_action_pairs):
        next_states = torch.zeros((len(s), self.state_size), device=self.device)

        # if the USAD returned true, then the state is unknown to our model, so we should return the absorbing state
        next_states[known_state_action_pairs] = self.absorbing_state

        # otherwise, just use the dynamics model to predict the next state
        if np.sum(known_state_action_pairs == False) > 0:
            # modified from: https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L47
            if type(s) == np.ndarray:
                s = torch.from_numpy(s).float().to(self.device)
            if type(a) == np.ndarray:
                a = torch.from_numpy(a).float().to(self.device)

            next_states[~known_state_action_pairs] = self.dynamics_model.predict(s[~known_state_action_pairs],
                                                                                 a[~known_state_action_pairs])

        return next_states

    def _forward_single(self, s, a, known_state_action_pair):
        # if the USAD returned true, then the state is unknown to our model, so we should return the absorbing state
        if known_state_action_pair:
            return self.absorbing_state
        # otherwise, just use the dynamics model to predict the next state
        else:
            # modified from: https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L47
            if type(s) == np.ndarray:
                s = torch.from_numpy(s).float().to(self.device)
            if type(a) == np.ndarray:
                a = torch.from_numpy(a).float().to(self.device)

            return self.dynamics_model.predict(s, a)

    def predict(self, s, a):
        # if the USAD returns true, then the state is unknown to our model, so we should return the absorbing state
        if self.usad(s, a):
            return self.absorbing_state.to('cpu').data.numpy()
        # otherwise, just use the dynamics model to predict the next state
        else:
            # modified from: https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L56
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            s_next = self.dynamics_model.predict(s, a)
            s_next = s_next.to('cpu').data.numpy()
            return s_next

    def reward(self, s, a):
        # this function only works when s and a are a batch of values
        # e.g., from s = trajectory['observation'], a = trajectory['action']
        if self.env_name == 'Ant-v2':
            r = reward_functions.ant_reward
        elif self.env_name == 'Hopper-v2':
            r = reward_functions.hopper_reward
        else:
            raise NotImplementedError(f'Reward function not implemented for environment: {self.env_name}')

        if not isinstance(s, np.ndarray):
            s = np.array(s)
        if not isinstance(a, np.ndarray):
            a = np.array(a)

        rewards = np.zeros(len(s))
        known_state_action_pairs = self.usad(s, a)

        # if any state matches the absorbing state, then we should return the negative reward
        # we can use all(axis=-1) to make sure that each state is being compared against the absorbing state
        # same thing if any of the state-action pairs are unknown (the USAD returned true),
        #   then we should return the negative reward
        negative_reward_locations = np.logical_or(known_state_action_pairs,
                                                  (s == self.absorbing_state.to('cpu').data.numpy()).all(axis=-1) )

        rewards[negative_reward_locations] = self.min_reward

        # otherwise, we can just use the reward function to get the reward for each state-action pair
        # note: the reward function expects s and a to be nonempty, so we need to make sure that there are some states
        #  and actions that are both known and not the absorbing state
        if np.sum(negative_reward_locations) >= 0:
            rewards[~negative_reward_locations] = r(s[~negative_reward_locations],
                                                   a[~negative_reward_locations])

        return rewards

    def compute_path_rewards(self, paths):
        # from https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L150
        # paths has two keys: observations and actions
        # paths["observations"] : (num_traj, horizon, obs_dim)
        # paths["rewards"] should have shape (num_traj, horizon)

        num_traj, horizon, _ = paths["observations"].shape
        rewards = np.zeros((num_traj, horizon))
        # for i in range(num_traj):
        #     for j in range(horizon):
        #         rewards[i, j] = self.reward(s[i, j], a[i, j])
        for num_traj, (state_batch, action_batch) in enumerate(zip(paths['observations'], paths['actions'])):
            rewards[num_traj] = self.reward(state_batch, action_batch)

        paths['rewards'] = rewards if rewards.shape[0] > 1 else rewards.ravel()

        return rewards

    def compute_loss(self, s, a, s_next):
        # taken from https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L79
        # Intended for logging use only, not for loss computation

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).float().to(self.device)
        #s_next = torch.from_numpy(s_next).float().to(self.device)

        sp = self.forward(s, a)
        s_next = torch.from_numpy(s_next).float() if type(s_next) == np.ndarray else s_next
        s_next = s_next.to(self.device)
        loss = self.loss_func(sp, s_next)
        return loss.to('cpu').data.numpy()

    def save(self, filename):
        torch.save(self.dynamics_model.state_dict(), filename)


if __name__ == '__main__':
    # running mdp.py allows us to quickly create and save an MDP model. Parameters are for Ant-v2 env
    # ant-v2 with the 10k step pure dataset:
    """python mdp.py --dataset dataset/TRPO_Ant-v2_10000 --env Ant-v2 --output trained_models/MDP_Ant-v2_10000 \
        --usad-output trained_models/USAD_Ant-v2_10000 --epochs 10
    """
    # hopper-v2 with 1 million step pure dataset:
    """
    python mdp.py --dataset dataset/TRPO_Hopper-v2_1000000 --env Hopper-v2 --output trained_models/MDP_Hopper-v2_1e6 \
        --negative-reward 50 --usad-output trained_models/USAD_Hopper-v2_1e6 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--usad-output', type=str, required=True)
    parser.add_argument('--negative-reward', type=float, default=100.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # load the dataset
    with open(args.dataset, 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)
        print(f'loaded dataset {args.dataset}')

    mdp = MDP(dataset, env_name=args.env, num_epochs=args.epochs,
              device=args.device, negative_reward=args.negative_reward)

    total_loss = 0
    total_loss_2 = 0
    for trajectory in dataset:
        for s, a, s_next in zip(trajectory['observations'], trajectory['actions'], trajectory['observations'][:1]):
            loss = mdp.compute_loss(s, a, s_next)
            total_loss += loss

        # s_batch = np.array(trajectory['observations'][:-1])
        # a_batch = np.array(trajectory['actions'][:-1])
        # s_next_batch = np.array(trajectory['observations'][1:])
        # total_loss_2 += mdp.compute_loss(s_batch, a_batch, s_next_batch)

    print('MDP Loss =', total_loss)
    #print('MDP Loss 2 =', total_loss_2)
    print('USAD Threshold =', mdp.usad.threshold)

    mdp.save(args.output)
    mdp.usad.save(args.usad_output)


