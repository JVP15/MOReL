import torch
import torch.nn as nn
import numpy as np

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, std = 0.01, device = 'cpu'):
        """This is the dynamics model from the MOReL algorithm. It is used by both the MDP and the USAD. It is equivalent
        to N(f(s,a), SIGMA) where f(s,a) = s + s_diff_std * MLP((s - s_mean) / s_std, (a - a_mean) / a_std)).
        The MLP uses 2 hidden layers with 512 neurons and RELU activation."""

        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, state_size)

        self.std = std

        self.state_mean = 0
        self.state_std = 0
        self.action_mean = 0
        self.action_std = 0
        self.state_difference_std = 0

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
        will be automatically converted to the device that the model is on"""

        self._init_statistics(dataset)

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