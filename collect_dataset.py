import numpy as np
import pickle

import gym
import argparse
from argparse import RawTextHelpFormatter

from sb3_contrib import TRPO

ACTION_NOISE_OPTIONS = ['pure', 'gauss', 'random']
DEFAULT_ACTION_NOISE = ['pure', 0.0] # the command line arg expects two values, but the 'pure' option doesn't need any extra values
DEFAULT_ITERATIONS = 1000000
DEFAULT_TRAJECTORY_LENGTH = 500
DEFAULT_DATASET_DIR = './dataset'
DEFAULT_VERBOSE = 0


NOISY_DATASET_COLLECT_PURE_PERCENTAGE = .4
NOISY_DATASET_COLLECT_RANDOM_PERCENT = .2


def sample_for_noisy_dataset(obs, policy, step, max_iterations, action_noise, action_noise_arg):
    # for 40% of the iterations, we sample from the pure policy
    if step < max_iterations * NOISY_DATASET_COLLECT_PURE_PERCENTAGE:
        action, _ = policy.predict(obs, deterministic=True)
    # for 20% of the iterations, we sample from the random policy
    elif step < max_iterations * (NOISY_DATASET_COLLECT_PURE_PERCENTAGE + NOISY_DATASET_COLLECT_RANDOM_PERCENT):
        action = env.action_space.sample()
    # for the remaining 40%, we sample using the pure policy with some noise
    # if action_noise = gauss, sample from the pure policy and apply gaussian noise with mean 0 and std to it
    elif action_noise == 'gauss':
        action, _ = policy.predict(obs, deterministic=True)
        action += np.random.normal(0, action_noise_arg, size=action.shape)
    # otherwise, action_noise = random, so sample from the random policy with probability q, and from the pure policy with probability 1-q
    else:
        if np.random.uniform() < action_noise_arg:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=True)

    return action


def collect_dataset(env, model, iterations, dataset_dir, action_noise, steps_per_trajectory, action_noise_arg=None):
    # the dataset is a collection of trajectories collected over many episodes of the environment
    # we don't collect a set number of trajectories; instead, we just run the agent in the environment for a number of
    #  steps equal to 'iterations' and record the observation, action, and reward at each step, then combine them
    #  into the trajectory for that episode

    trajectories = []

    observations = []
    actions = []
    rewards = []

    print('Collecting dataset...')
    obs = env.reset()
    step_num = 0
    for i in range(iterations):

        if action_noise == 'pure':
            action, _state = model.predict(obs, deterministic=True)
        else:
            action = sample_for_noisy_dataset(obs, model, i, iterations, action_noise, action_noise_arg)

        next_obs, reward, done, _info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs
        step_num += 1

        # if the episode is done, reset the environment, collect the observations, actions, and rewards into a
        #  single trajectory, and reset the environment for the next episode
        if done or step_num == steps_per_trajectory:
            obs = env.reset()
            trajectories.append(dict(observations=observations, actions=actions, rewards=rewards))

            # empty the observations, actions, and rewards lists for the next episode
            observations = []
            actions = []
            rewards = []
            step_num = 0

        if i % 1000 == 0:  # TODO: add verbose command line argument and make better progress bar
            print(f'\rCollecting Dataset. Currently on step {i}/{iterations}', end='', flush=True)

    # get the name of the environment and the policy used to collect the dataset (for saving purposes)
    env_name = env.unwrapped.spec.id
    model_name = model.__class__.__name__
    dataset_file_name = f'{dataset_dir}/{model_name}_{env_name}_{iterations}'

    if action_noise != 'pure':
        dataset_file_name += f'_{action_noise}_{action_noise_arg}'

    # save the dataset to a file
    with open(dataset_file_name, 'wb') as dataset_file:
        pickle.dump(trajectories, dataset_file)

    print(f'\nDataset saved to {dataset_file_name}')

if __name__ == '__main__':
    # example usage: python collect_dataset.py --env Ant-v2 --policy trained_policies/TRPO_Ant-v2_1000.zip --dir ./dataset
    # python collect_dataset.py --env Hopper-v2 --policy trained_policies/TRPO_Hopper-v2_1000.zip --dir ./dataset --iterations 1000000 --trajectory-length 400

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--env', type=str, required=True, help='<Required> environment to run')
    parser.add_argument('--policy', type=str, required=True, help='<Required> location of the policy file')
    # unfortunately, I couldn't figure out a clean way to save this description
    parser.add_argument('--action-noise', type=str, default=DEFAULT_ACTION_NOISE, nargs=2, help="""whether the dataset will be generated with action noise or not. There are three options: 'pure', 'gauss', and 'random'.
 pure: (default) the dataset is collected using the specified policy P. Note: this option does not require any additional arguments, but the command line interface will still expect two arguments, so you must specify a value for the second argument.
 gauss: usage --action-noise gauss std
    std 40%% of the dataset is collected using P. 
    40%% of the dataset is collected using zero-mean Gaussian noise with standard deviation std added to the actions sampled from P.'
    20%% of the dataset is collected using a uniform random policy
 random: usage --action-noise random q 
    40%% of the dataset is collected using P. 
    40%% of the dataset is collected using a policy that samples from P with probability 1-q and samples from a uniform random policy with probability q.
    20%% of the dataset is collected using a uniform random policy""")

    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS, help=f'number of iterations to save (default: {DEFAULT_ITERATIONS})')
    parser.add_argument('--trajectory-length', type=int, default=DEFAULT_TRAJECTORY_LENGTH, help=f'number of steps to take when collecting each trajectory (default: {DEFAULT_TRAJECTORY_LENGTH}')
    parser.add_argument('--dir', type=str, default=DEFAULT_DATASET_DIR, help=f'directory to save dataset (default: {DEFAULT_DATASET_DIR})')
    args = parser.parse_args()

    # check to make sure that the action noise argument is valid
    if args.action_noise[0].lower() not in ACTION_NOISE_OPTIONS:
        raise ValueError(f'Invalid action noise option: {args.action_noise}. Valid options are {ACTION_NOISE_OPTIONS}')

    env = gym.make(args.env)
    model = TRPO.load(args.policy, env=env)

    collect_dataset(env=env, model=model, iterations=args.iterations, dataset_dir=args.dir, steps_per_trajectory=args.trajectory_length,
                    action_noise=args.action_noise[0], action_noise_arg=float(args.action_noise[1]))

