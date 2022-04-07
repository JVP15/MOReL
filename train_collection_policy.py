import gym
import argparse

from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

DEFAULT_ENV = 'CartPole-v1'
DEFAULT_TARGET_REWARD = 450
DEFAULT_POLICY_DIR = './trained_policies'
DEFAULT_MAX_ITERATIONS = 1000000
DEFAULT_EVAL_INTERVAL = 1000
DEFAULT_EVAL_EPISODES = 5
DEFAULT_VERBOSE = 0

def train_collection_policy(env_name, target_reward, policy_dir,max_training_iterations,
                            eval_interval, eval_episodes, verbose):
    """
    Train a collection policy for the given environment until the target reward is reached.

    :param env_name: the environment name (e.g. "Hopper-v2")
    :param target_reward: the target reward for the policy to stop training
    :param policy_dir: directory to save the trained policies
    :param max_training_iterations: maximum number of training iterations
    :param eval_interval: interval in which to evaluate the policy
    :param eval_episodes: Number of episodes to evaluate the policy on.
    :param verbose: verbosity level 0: no output, 1: output every evaluation, 2: debug output every evaluation
    """

    # Create the environment
    env = gym.make(env_name)
    eval_env = Monitor(gym.make(env_name))

    # Create the model
    model = TRPO('MlpPolicy', env, verbose=0)

    # Create the callback
    reward_threshold_callback = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=verbose)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=reward_threshold_callback, verbose=verbose,
                                 n_eval_episodes=eval_episodes, eval_freq=eval_interval)

    # Train the model
    model.learn(total_timesteps=max_training_iterations, callback=eval_callback)

    # Save the model
    model.save(f'{policy_dir}/TRPO_{env_name}_{target_reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default=DEFAULT_ENV)
    parser.add_argument('--target-reward', help='target reward', default=DEFAULT_TARGET_REWARD, type=int)
    parser.add_argument('--policy-dir', help='save policy directory', default=DEFAULT_POLICY_DIR)
    parser.add_argument('--max-training-iterations', help='max training iterations', default=DEFAULT_MAX_ITERATIONS, type=int)
    parser.add_argument('--eval-interval', help='evaluation interval', default=DEFAULT_EVAL_INTERVAL, type=int)
    parser.add_argument('--eval-episodes', help='number of evaluation episodes', default=DEFAULT_EVAL_EPISODES, type=int)
    parser.add_argument('--verbose', help='verbosity', default=DEFAULT_VERBOSE, type=int)
    args = parser.parse_args()

    train_collection_policy(args.env, args.target_reward, args.policy_dir, args.max_training_iterations,
                            args.eval_interval, args.eval_episodes, args.verbose)
