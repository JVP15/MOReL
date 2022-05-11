import time

import argparse

import torch
import torch.nn as nn
import numpy as np
import pickle

def ant_reward(s,a):
    # mimics reward function for the ant-v2 environment from:
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py (for values)
    # and
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py
    # for information about what each value in the state represents
    # for some reason, it isn't the exact same reward, but it is close enough, so we'll use it
    healthy_reward = 1.0
    x_velocity = s[13] # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py#L69
    forward_reward = x_velocity

    rewards = healthy_reward + forward_reward

    contact_force = s[27:] # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py#L85
    contact_cost_weight = 1e-3
    contact_cost = contact_cost_weight * np.sum(np.square(np.clip(contact_force, -1, 1)))
    control_cost_weight = .5
    control_cost = control_cost_weight * np.sum(np.square(a))

    costs = contact_cost + control_cost
    reward = rewards - costs

    return reward

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
                 model_path=None):
        """Models a pessimistic MDP for the MOReL algorithm.
        :param dataset: The dataset to use for training. It is expected to be a list of trajectories where each
        trajectory is a dictionary of the form {'observations': [], 'actions': [], 'rewards': []}.

        :param model_path: The path to a saved dynamics model. NOTE: if you are loading a model, you should load the
        model should have been trained using the dataset provided, otherwise the MDP statistics (mean and std for
        state and actions) could be differen than the statistics that were used to train the saved dynamics model.

        Interface for this class is (mostly) taken from the WorldModel class from MBRL:
        https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L7"""
        self.env_name = env_name
        self.device = device

        self.std = std

        self.state_mean = 0
        self.state_std = 0
        self.action_mean = 0
        self.action_std = 0
        self.state_difference_std = 0

        self.action_size = 0
        self.state_size = 0

        # for simplicity's sake, we'll just set the absorbing state to be all 0s. I'm not sure if this is how they
        #   actually did it in the MOReL paper, but it is what I am going with for now.
        self.absorbing_state = torch.zeros(self.state_size)
        self.absorbing_state.to(device)

        # this is from the WorldModel class. Even though we don't 'technically' learn a rewards network,
        #   if we want ModelBasedNPG to use our USAD-based reward function, we need to set this to true
        self.learn_reward = True
        self.min_reward = np.inf

        # for the time being, just say that every state is known
        self.usad = lambda s, a: False

        self._init_statistics(dataset)
        self.negative_reward = self.min_reward - negative_reward

        dynamics_model_args = [self.state_size, self.action_size,
                               self.state_mean, self.state_std,
                               self.action_mean, self.action_std,
                               self.state_difference_std,
                               std,
                               device]

        self.dynamics_model = DynamicsModel(*dynamics_model_args)
        if model_path:
            print(f'MDP: Loading dynamics model from {model_path}')
            self.dynamics_model.load_state_dict(torch.load(model_path))
        else:
            self.dynamics_model.fit(dataset, num_epochs, batch_size, learning_rate)


    def _init_statistics(self, dataset):
        all_actions = []
        all_states = []
        all_state_diffs = []

        for trajectory in dataset:
            all_actions.extend(trajectory['actions'])
            all_states.extend(trajectory['observations'])

            for i in range(len(trajectory['observations']) - 1):
                all_state_diffs.append(trajectory['observations'][i + 1] - trajectory['observations'][i])

            if np.min(trajectory['rewards']) < self.min_reward:
                self.min_reward = np.min(trajectory['rewards'])

        self.state_mean = np.mean(all_states)
        self.state_std = np.std(all_states)
        self.action_mean = np.mean(all_actions)
        self.action_std = np.std(all_actions)
        self.state_difference_std = np.std(all_state_diffs)

        self.action_size = len(all_actions[0])
        self.state_size = len(all_states[0])

    def to(self, device):
        self.dynamics_model.to(device)
        self.absorbing_state.to(device)

    def is_cuda(self):
        return self.dynamics_model.device.startswith('cuda')

    def forward(self, s, a):
        # if the USAD returns true, then the state is unknown to our model, so we should return the absorbing state
        if self.usad(s, a):
            return self.absorbing_state
        # otherwise, just use the dynamics model to predict the next state
        else:
            # modified from: https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L47
            if type(s) == np.ndarray:
                s = torch.from_numpy(s).float()
            if type(a) == np.ndarray:
                a = torch.from_numpy(a).float()

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
        if self.env_name == 'Ant-v2':
            r = ant_reward
        else:
            raise NotImplementedError(f'Reward function not implemented for environment: {self.env_name}')

        if np.array_equal(s, self.absorbing_state) or self.usad(s, a):
            return self.negative_reward
        else:
            return r(s, a)

    def compute_path_rewards(self, paths):
        # from https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L150
        # paths has two keys: observations and actions
        # paths["observations"] : (num_traj, horizon, obs_dim)
        # paths["rewards"] should have shape (num_traj, horizon)
        s, a = paths['observations'], paths['actions']
        num_traj, horizon, _ = s.shape

        rewards = np.zeros((num_traj, horizon))
        for i in range(num_traj):
            for j in range(horizon):
                rewards[i, j] = self.reward(s[i, j], a[i, j])

        paths['rewards'] = rewards

    def compute_loss(self, s, a, s_next):
        # taken from https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L79
        # Intended for logging use only, not for loss computation

        sp = self.forward(s, a)
        s_next = torch.from_numpy(s_next).float() if type(s_next) == np.ndarray else s_next
        s_next = s_next.to(self.device)

        loss_func = nn.MSELoss()
        loss = loss_func(sp, s_next)
        return loss.to('cpu').data.numpy()

    def save(self, filename):
        torch.save(self.dynamics_model.state_dict(), filename)

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, state_mean, state_std, action_mean, action_std, state_difference_std, std = 0.01, device = 'cpu'):
        """This is the dynamics model from the MOReL algorithm. It is used by both the MDP and the USAD. It is equivalent
        to N(f(s,a), SIGMA) where f(s,a) = s + s_diff_std * MLP((s - s_mean) / s_std, (a - a_mean) / a_std)).
        The MLP uses 2 hidden layers with 512 neurons and RELU activation."""

        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, state_size)

        self.std = std

        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_difference_std = state_difference_std

        self.device = device
        self.to(device)

    def forward(self, s, a):
        # equivalent to f(s, a) in the paper
        # f(s,a) = s + s_diff_std * MLP((s - s_mean) / s_std, (a - a_mean) / a_std))

        s_normalized = (s - self.state_mean) / self.state_std
        a_normalized = (a - self.action_mean) / self.action_std

        # make sure that the tensors are on the same device
        s_normalized = s_normalized.to(self.device)
        a_normalized = a_normalized.to(self.device)

        # concatenate s and a to create the input to the nn
        mu = torch.cat((s_normalized, a_normalized), dim=-1)

        mu = self.fc1(mu)
        mu = torch.relu(mu)
        mu = self.fc2(mu)
        mu = torch.relu(mu)
        mu = self.fc3(mu)

        return mu

    def predict(self, s, a):
        # equivalent to N(f(s,a), SIGMA) in the paper where f(s,a) is given by forward
        s = s.to(self.device)
        out = s + torch.normal(self.forward(s, a), self.std)
        return out

    def fit(self, dataset, num_epochs = 300, batch_size = 256, learning_rate = 5e-4):
        """Trains the dynamics model using the given dataset.
        :param dataset: The dataset to use for training. It is expected to be a list of trajectories where each
        trajectory is a dictionary of the form {'observations': []], 'actions': [], 'rewards': []}. Each list
        will automatically be automatically converted to the device that the model is on"""

        # convert the dataset into a single list of (s, a, s')
        dataset = [(s, a, s_prime) for trajectory in dataset for s, a, s_prime in zip(trajectory['observations'], trajectory['actions'], trajectory['observations'][1:])]

        # shuffle the dataset
        np.random.shuffle(dataset)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        print('Training dynamics model...')

        for epoch in range(num_epochs):
            print(f'\rTraining Model. Currently on epoch {epoch}/{num_epochs}', end='', flush=True)

            for i in range(0, len(dataset), batch_size):
                # get a minibatch of data
                batch = dataset[i:i + batch_size]
                # separate the list of s, a, and s' out of the batch of data. We have to convert them to tensors and
                #  put them on the same device as the model before we can start training with them. We also have to
                #  convert them to floats because the model expects them to be floats not Doubles

                s_batch = np.array([data[0] for data in batch])
                a_batch = np.array([data[1] for data in batch])
                s_prime_batch = np.array([data[2] for data in batch])

                s_batch = torch.tensor(s_batch, device=self.device, dtype=torch.float)
                a_batch = torch.tensor(a_batch, device=self.device, dtype=torch.float)
                s_prime_batch = torch.tensor(s_prime_batch, device=self.device, dtype=torch.float)

                #optimizer.zero_grad()
                for param in self.parameters():
                    param.grad = None

                # get the next state predictions
                next_states = self.predict(s_batch, a_batch)

                # compute the loss
                loss = loss_fn(next_states, s_prime_batch)

                # backpropagate the loss

                loss.backward()
                optimizer.step()

        print('\nTrained Model')

        return self

    def to(self, device):
        super().to(device)
        self.device = device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--negative-reward', type=float, default=100.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # load the dataset
    with open(args.dataset, 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)
        print(f'loaded dataset {args.dataset}')

    mdp = MDP(dataset, num_epochs=args.num_epochs, env_name=args.env,
              device=args.device, negative_reward=args.negative_reward)
    print(mdp.state_mean)
    print(mdp.state_std)
    print(mdp.action_mean)
    print(mdp.action_std)
    print(mdp.state_difference_std)

    mdp.save(args.output)

