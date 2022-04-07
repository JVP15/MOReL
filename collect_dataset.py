import pickle

import gym
import argparse
from argparse import RawTextHelpFormatter

from sb3_contrib import TRPO

DEFAULT_ACTION_NOISE = 'pure'
DEFAULT_ITERATIONS = 1000000
DEFAULT_DATASET_DIR = './dataset'
DEFAULT_VERBOSE = 0

def collect_dataset(env, policy, iterations, dataset_dir, action_noise, **action_noise_kwargs):
    # the dataset is a collection of trajectories collected over many episodes of the environment
    # we don't collect a set number of trajectories; instead, we just run the agent in the environment for a number of
    #  steps equal to 'iterations' and record the observation, action, and reward at each step, then combine them
    #  into the trajectory for that episode

    trajectories = []
    print('Collecting dataset...')
    obs = env.reset()
    for i in range(iterations):
        observations = []
        actions = []
        rewards = []

        action, _state = model.predict(obs, deterministic=True)
        # TODO: add support for action noise. Right now it only supports the pure dataset collection
        obs, reward, done, _info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        # if the episode is done, reset the environment, collect the observations, actions, and rewards into a
        #  single trajectory, and reset the environment for the next episode
        if done:
            obs = env.reset()
            trajectories.append(dict(observations=observations, actions=actions, rewards=rewards))

            # empty the observations, actions, and rewards lists for the next episode
            observations.clear()
            actions.clear()
            rewards.clear()

        if i % 1000 == 0:  # TODO: add verbose command line argument and make better progress bar
            print(f'\rCollecting Dataset. Currently on step {i}/{iterations}', end='', flush=True)

    # get the name of the environment and the policy used to collect the dataset (for saving purposes)
    env_name = env.unwrapped.spec.id
    policy_name = policy.__class__.__name__
    dataset_file_name = f'{dataset_dir}/{policy_name}_{env_name}_{iterations}'

    # save the dataset to a file
    with open(dataset_file_name, 'wb') as dataset_file:
        pickle.dump(trajectories, dataset_file)

    print(f'\nDataset saved to {dataset_file_name}')

if __name__ == '__main__':
    # example usage: python collect_dataset.py --env Ant-v2 --policy trained_policies/TRPO_Ant-v2_1000.zip --dir ./dataset

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--env', type=str, required=True, help='<Required> environment to run')
    parser.add_argument('--policy', type=str, required=True, help='<Required> location of the policy file')
    # unfortunately, I couldn't figure out a clean way to save this description
    parser.add_argument('--action-noise', type=str, default=DEFAULT_ACTION_NOISE, nargs=2, help="""whether the dataset will be generated with action noise or not. There are three options: 'pure', 'gauss', and 'random'.
 pure: (default) the dataset is collected using the specified policy P.
 gauss: usage --action-noise gauss 
    std 40%% of the dataset is collected using P. 
    40%% of the dataset is collected using zero-mean Gaussian noise with standard deviation std added to the actions sampled from P.'
    20%% of the dataset is collected using a uniform random policy
 random: usage --action-noise random q 
    40%% of the dataset is collected using P. 
    40%% of the dataset is collected using a policy that samples from P with probability 1-q and samples from a uniform random policy with probability q.
    20%% of the dataset is collected using a uniform random policy""")

    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS, help=f'number of iterations to save (default: {DEFAULT_ITERATIONS})')
    parser.add_argument('--dir', type=str, default=DEFAULT_DATASET_DIR, help=f'directory to save dataset (default: {DEFAULT_DATASET_DIR})')
    args = parser.parse_args()

    env = gym.make(args.env)
    model = TRPO.load(args.policy, env=env)

    collect_dataset(env=env, policy=model, iterations=args.iterations, dataset_dir=args.dir, action_noise=args.action_noise)

