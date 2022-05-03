import torch
import torch.nn as nn
import numpy as np
import pickle

class MDP(object):
    def __init__(self, dataset, negative_reward = 100, std = 0.01, num_epochs = 300, batch_size = 256, learning_rate = 5e-4, device='cpu'):
        """Models a pessimistic MDP for the MOReL algorithm.
        :param dataset: The dataset to use for training. It is expected to be a list of trajectories where each
        trajectory is a dictionary of the form {'observations': [], 'actions': [], 'rewards': []}.

        Interface for this class is (mostly) taken from the WorldModel class from MBRL:
        https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L7"""

        self.std = std

        self.state_mean = 0
        self.state_std = 0
        self.action_mean = 0
        self.action_std = 0
        self.state_difference_std = 0

        self.action_size = 0
        self.state_size = 0

        self.negative_reward = 100
        # this is from the WorldModel class. Even though we don't 'technically' learn a rewards network,
        #   if we want ModelBasedNPG to use our USAD-based reward function, we need to set this to true
        self.learn_reward = True
        self.min_reward = np.inf

        # for the time being, just say that every state is known
        self.usad = lambda s, a: False

        self._init_statistics(dataset)
        dynamics_model_args = [self.state_size, self.action_size,
                               self.state_mean, self.state_std,
                               self.action_mean, self.action_std,
                               self.state_difference_std,
                               std,
                               device]
        self.dynamics_model = DynamicsModel(*dynamics_model_args)
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

    def is_cuda(self):
        return self.dynamics_model.device.startswith('cuda')

    def forward(self, s, a):
        # modified from: https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L47
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()
        return self.dynamics_model.forward(s, a)

    def predict(self, s, a):
        # modified from: https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L56
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s_next = self.dynamics_model.forward(s, a)
        s_next = s_next.to('cpu').data.numpy()
        return s_next

    def compute_path_rewards(self, paths):
        # from https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L150
        # paths has two keys: observations and actions
        # paths["observations"] : (num_traj, horizon, obs_dim)
        # paths["rewards"] should have shape (num_traj, horizon)
        s, a, r = paths['observations'], paths['actions'], paths['rewards']




class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, state_mean, state_std, action_mean, action_std, state_difference_std, std = 0.01, device = 'cpu'):
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


    def forward(self, s, a):
        # equivalent to f(s, a) in the paper
        # f(s,a) = s + s_diff_std * MLP((s - s_mean) / s_std, (a - a_mean) / a_std))

        s_normalized = (s - self.state_mean) / self.state_std
        a_normalized = (a - self.action_mean) / self.action_std

        # concatenate s and a to create the input to the nn
        mu = torch.cat((s_normalized, a_normalized), dim=-1).to(self.device)

        mu = self.fc1(mu)
        mu = torch.relu(mu)
        mu = self.fc2(mu)
        mu = torch.relu(mu)
        mu = self.fc3(mu)

        return mu

    def sample(self, s, a):
        # equivalent to N(f(s,a), SIGMA) in the paper where f(s,a) is given by forward
        out = s + torch.normal(self.forward(s, a), self.std)
        return out

    def fit(self, dataset, num_epochs = 300, batch_size = 256, learning_rate = 5e-4):
        """Trains the dynamics model using the given dataset.
        :param dataset: The dataset to use for training. It is expected to be a list of trajectories where each
        trajectory is a dictionary of the form {'observations': []], 'actions': [], 'rewards': []}. Each list
        will automatically be automatically converted to the device that the model is on"""

        # shuffle the dataset. Thanks to Stack Overflow user sshashank124 for the code to shuffle multiple lists from
        #   their answer here: https://stackoverflow.com/a/23289591
        # dataset = list(zip(dataset['observations'],
        #                    dataset['actions'],
        #                    dataset['observations'][1:]))
        # np.random.shuffle(dataset)
        # s, a, s_prime = zip(*dataset)
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
                #  convert them to floats for some reason (it throws an error if I don't)
                s_batch = np.array([data[0] for data in batch])
                a_batch = np.array([data[1] for data in batch])
                s_prime_batch = np.array([data[2] for data in batch])
                
                s_batch = torch.tensor(s_batch).float().to(self.device)
                a_batch = torch.tensor(a_batch).float().to(self.device)
                s_prime_batch = torch.tensor(s_prime_batch).float().to(self.device)

                optimizer.zero_grad()

                # get the next state predictions
                next_states = self.sample(s_batch, a_batch)

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

    # load the dataset
    with open('dataset/TRPO_Ant-v2_50000', 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)
        print('loaded dataset')

    mdp = MDP(dataset)
    print(mdp.state_mean)
    print(mdp.state_std)
    print(mdp.action_mean)
    print(mdp.action_std)
    print(mdp.state_difference_std)

    f = DynamicsModel(mdp.state_size, mdp.action_size, mdp.state_mean, mdp.state_std, mdp.action_mean, mdp.action_std, mdp.state_difference_std)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f.to(device)
    f.fit(dataset, num_epochs=50)