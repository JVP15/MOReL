import numpy as np
import torch
import os
from dynamics_model import DynamicsModel

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

        self.batch_size = batch_size

        self.dynamics_models = []

        if usad_folder is not None:
            print(f'USAD: loading dynamics models from {usad_folder}')
            self._load_dynamics_models(dataset, usad_folder)
        else:
            print('Training USAD...')
            self._train_dynamics_models(dataset, num_epochs, batch_size, learning_rate)

        self.threshold = 0

        self._find_threshold(dataset)

    def __call__(self, s, a):
        """This is the U_practical(s,a) = {False  if disc(s,a) <= threshold,
                                          {True   if disc(s,a) > threshold
        function from the MOReL paper. It has been modified to work with batches of states and actions"""

        discrepancies = self.disc(s,a)

        # if an element is false, the state is known. If it is true, the state is unknown
        states_actions_known = discrepancies > self.threshold

        return states_actions_known

    def disc(self, s, a):
        """This is the disc(s, a) = max_ij ||f_i(s,a) - f_j(s,a)|| from the MOReL paper.
        It has been modified to work with batches of states and actions"""

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
            model = DynamicsModel(dataset_split[i], device=self.device)

            # train the dynamics model on one part of the dataset
            model.fit(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
            self.dynamics_models.append(model)

    def _load_dynamics_models(self, dataset, model_folder):
        """Loads pre-trained dynamics models from a folder. Note: the folder must *only* contain
        pre-trained dynamics models"""
        filenames = os.listdir(model_folder)

        for model_filename in filenames:
            model_path = os.path.join(model_folder, model_filename)
            model = DynamicsModel(dataset, device=self.device)
            model.load_state_dict(torch.load(model_path))
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