import os

import time

import argparse

import torch
import torch.nn as nn
import numpy as np
import pickle
import reward_functions
from dynamics_model import DynamicsModel


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

        :param model_path: The path to a saved dynamics model. NOTE: if you are loading a model, you should load the
        model should have been trained using the dataset provided, otherwise the MDP statistics (mean and std for
        state and actions) could be differen than the statistics that were used to train the saved dynamics model.
        :param usad_folder: The path to a folder containing pretrained dynamics models for the USAD.

        Interface for this class is (mostly) taken from the WorldModel class from MBRL:
        https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py#L7"""

        # set the environment for the reward function. We're expecting a GymEnv from MJRL, but we only need the gym.Env object that it wraps
        self.env_name = env_name

        self.device = device

        self.std = std

        self.action_size = 0
        self.state_size = 0

        # this is from the WorldModel class. It tells the planning algorithm whether the MDP uses a learned
        #  rewards function or if it uses the true reward function from the environment
        self.learn_reward = False
        self.min_reward = np.inf

        # for simplicity's sake, we'll just set the absorbing state to be all 0s. I'm not sure if this is how they
        #   actually did it in the MOReL paper, but it is what I am going with for now.
        self.absorbing_state = torch.zeros(self.state_size)
        self.absorbing_state.to(device)

        self._init_statistics(dataset)
        self.negative_reward = self.min_reward - negative_reward

        self.dynamics_model = DynamicsModel(self.state_size, self.action_size, std, device)
        if model_path:
            print(f'MDP: Loading dynamics model from {model_path}')
            self.dynamics_model.load_state_dict(torch.load(model_path))
        else:
            self.dynamics_model.fit(dataset, num_epochs, batch_size, learning_rate)

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
            r = reward_functions.ant_reward
        elif self.env_name == 'Hopper-v2':
            r = reward_functions.hopper_reward
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
        loss = self.loss_func(sp, s_next)
        return loss.to('cpu').data.numpy()

    def save(self, filename):
        torch.save(self.dynamics_model.state_dict(), filename)


class USAD(object):
    def __init__(self,
                 dataset,
                 state_size,
                 action_size,
                 num_models=4,
                 num_epochs=300,
                 batch_size=256,
                 learning_rate=5e-4,
                 device='cpu',
                 usad_folder=None):
        """Models the Unknown State Detector (USAD) for the MOReL algorithm.
        :param usad_folder: a folder that contains pre-trained dynamics models. Note: this folder must
        *only* contain pre-trained dynamic models"""

        self.num_models = num_models
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.dynamics_models = []

        if usad_folder is not None:
            print(f'USAD: loading dynamics models from {usad_folder}')
            self._load_dynamics_models(usad_folder)
        else:
            print('Training USAD...')
            self._train_dynamics_models(dataset, num_epochs, batch_size, learning_rate)

        self.threshold = 0

        self._find_threshold(dataset)

    def __call__(self, s, a):
        """This is the U_practical(s,a) = {False  if disc(s,a) <= threshold,
                                          {True   if disc(s,a) > threshold
        function from the MOReL paper"""

        max_disc = self.disc(s,a)

        if max_disc <= self.threshold:
            return False # The state is known
        else:
            return True # the state is unknown

    def disc(self, s, a):
        """This is the disc(s, a) = max_ij ||f_i(s,a) - f_j(s,a)|| from the MOReL paper"""

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.array(a), dtype=torch.float32, device=self.device)

        diffs = []
        for i, model_i in enumerate(self.dynamics_models):
            for j in range(i+1, self.num_models):
                model_j = self.dynamics_models[j]
                # model.forward is f(s,a), whereas model.predict is N(f(s,a), SIGMA), so we use model.forward here
                difference = model_i.forward(s, a) - model_j.forward(s, a)

                diffs.append(torch.linalg.vector_norm(difference, dim=-1))

        return torch.max(torch.stack(diffs), dim=0).values.cpu().data.numpy()

    def save(self, usad_folder):
        if not os.path.exists(usad_folder):
            os.makedirs(usad_folder)

        for i, model in enumerate(self.dynamics_models):
            filename = os.path.join(usad_folder, f'USAD_model_{i}')
            torch.save(model.state_dict(), filename)


    def _train_dynamics_models(self, dataset, num_epochs, batch_size, learning_rate):
        # before training the dynamics models, we need to shuffle the dataset and split it into equal parts
        np.random.shuffle(dataset)
        dataset_split = np.array_split(dataset, self.num_models)

        for i in range(self.num_models):
            print(f'USAD: training dynamics model {i}')
            model = DynamicsModel(state_size=self.state_size, action_size=self.action_size, device=self.device)

            # train the dynamics model on one part of the dataset
            model.fit(dataset_split[i], num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
            self.dynamics_models.append(model)

    def _load_dynamics_models(self, model_folder):
        """Loads pre-trained dynamics models from a folder. Note: the folder must *only* contain
        pre-trained dynamics models"""
        filenames = os.listdir(model_folder)

        for model_filename in filenames:
            model_path = os.path.join(model_folder, model_filename)
            model = DynamicsModel(state_size=self.state_size, action_size=self.action_size, device=self.device)
            model.load_state_dict(torch.load(model_filename))
            self.dynamics_models.append(model)

        self.num_models = len(self.dynamics_models)

    def _find_threshold(self, dataset):
        disc_values = []

        for trajectory in dataset:
            s = trajectory["observations"]
            a = trajectory["actions"]
            disc_values.extend(self.disc(s, a))

        disc_mean = np.mean(disc_values)
        disc_std = np.std(disc_values)
        disc_max = np.max(disc_values)
        beta_max = (disc_max - disc_mean) / disc_std
        self.threshold = disc_mean + beta_max * disc_std

if __name__ == '__main__':
    # running mdp.py allows us to quickly create and save an MDP model. Parameters are for Ant-v2 env
    # ant-v2 with the 10k step pure dataset:
    # python mdp.py --dataset dataset/TRPO_Ant-v2_10000 --env Ant-v2 --output trained_models/MDP_Ant-v2_10000 --usad-output trained_models/USAD_Ant-v2_10000 --epochs 10
    # hopper-v2 with 1 million step pure dataset:
    # python mdp.py --dataset dataset/TRPO_Hopper-v2_1000000 --env Hopper-v2 --output trained_models/MDP_Hopper-v2_1e6 --negative-reward 50

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
    for trajectory in dataset:
        for s, a, s_next in zip(trajectory['observations'], trajectory['actions'], trajectory['observations'][:1]):
            loss = mdp.compute_loss(s, a, s_next)
            total_loss += loss

    print('MDP Loss =', total_loss)
    print('USAD Threshold =', mdp.usad.threshold)
    mdp.save(args.output)
    mdp.usad.save(args.usad_output)
