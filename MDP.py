import torch
import torch.nn as nn
import numpy as np
import pickle

class MDP(object):
    def __init__(self, dataset, std = 0.1, num_epochs = 300, batch_size = 256, learning_rate = 5e-4):
        """Models a pessimistic MDP for the MOReL algorithm.
        :param dataset: The dataset to use for training. It is expected to be a list of trajectories where each
        trajectory is a dictionary of the form {'observations': [], 'actions': [], 'rewards': []}."""

        self.std = std

        self.state_mean = 0
        self.state_std = 0
        self.action_mean = 0
        self.action_std = 0
        self.state_difference_std = 0

        self.action_size = 0
        self.state_size = 0

        # for the time being, just say that every state is known
        self.usad = lambda s, a: False

        self._init_statistics(dataset)

    def _init_statistics(self, dataset):
        all_actions = []
        all_states = []
        all_state_diffs = []

        for trajectory in dataset:
            all_actions.extend(trajectory['actions'])
            all_states.extend(trajectory['observations'])

            for i in range(len(trajectory['observations']) - 1):
                all_state_diffs.append(trajectory['observations'][i + 1] - trajectory['observations'][i])

        self.state_mean = np.mean(all_states)
        self.state_std = np.std(all_states)
        self.action_mean = np.mean(all_actions)
        self.action_std = np.std(all_actions)
        self.state_difference_std = np.std(all_state_diffs)

        self.action_size = len(all_actions[0])
        self.state_size = len(all_states[0])

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, state_mean, state_std, action_mean, action_std, state_difference_std, std = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, state_size)

        self.std = std

        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_difference_std = state_difference_std


    def forward(self, s, a):

        # concatenate s and a to create the input to the nn
        mu = torch.cat((s, a), dim=-1)

        mu = self.fc1(mu)
        mu = torch.relu(mu)
        mu = self.fc2(mu)
        mu = torch.relu(mu)
        mu = self.fc3(mu)

        out = torch.normal(mu, self.std)
        return out

    def fit(self, dataset, num_epochs = 300, batch_size = 256, learning_rate = 5e-4):
        """Trains the dynamics model using the given dataset.
        :param dataset: The dataset to use for training. It is expected to be a list of trajectories where each
        trajectory is a dictionary of the form {'observations': [], 'actions': [], 'rewards': []}."""

        # convert the dataset into a single list of (s, a, s') tuples
        dataset = [(s, a, s_prime) for trajectory in dataset for s, a, s_prime in zip(trajectory['observations'], trajectory['actions'], trajectory['observations'][1:])]
        # shuffle the dataset
        np.random.shuffle(dataset)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]


                optimizer.zero_grad()

                # get the next state predictions
                #next_states = self.forward(states, actions)

                # compute the loss
                loss = loss_fn()

                # backpropagate the loss

                loss.backward()
                optimizer.step()

        return self

if __name__ == '__main__':

    # load the dataset
    with open('dataset/TRPO_Ant-v2_1000000', 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)
        print('loaded dataset')

    mdp = MDP(dataset)
    print(mdp.state_mean)
    print(mdp.state_std)
    print(mdp.action_mean)
    print(mdp.action_std)
    print(mdp.state_difference_std)