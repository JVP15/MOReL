import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
from tabulate import tabulate

import mjrl.envs
import time as timer
import argparse
import os
import json
import mjrl.samplers.core as sampler
import mjrl.utils.tensor_utils as tensor_utils
from tqdm import tqdm


from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.logger import DataLog
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
from mjrl.algos.mbrl.sampling import sample_paths, evaluate_policy

from mdp import MDP

DEFAULT_OUTPUT_DIR = 'output'
DEFAULT_DEVICE = 'cuda'
DEFAULT_MODEL_PATH = None

DEFAULT_EVAL_ROLLOUTS = 4
DEFAULT_SAVE_FREQ = 25

# default values are for Ant-v2
DEFAULT_TRAINING_EPOCHS = 300
DEFAULT_NEGATIVE_REWARD = 100

DEFAULT_NPG_UPDATES = 1000
DEFAULT_HORIZON = 500
DEFAULT_INIT_LOG_STD = -.01
DEFAULT_MIN_LOG_STD = -2
DEFAULT_NUM_GRADIENT_TRAJECTORIES = 200
DEFAULT_CG_STEPS = 10

def run_morel(env_name, dataset, model_path, model_save_path, output_dir, device, dynamics_model_training_epochs, negative_reward,
              eval_rollouts, save_freq, npg_kwargs):
    """Implementation of the MOReL algorithm from: https://arxiv.org/pdf/2005.05951.pdf
    :param env_name: name of the environment to run MOReL on
    :param dataset: the dataset to use for training the MDP and dynamics models. It should be a list of trajectories
    where each trajectory is a dictionary of the form {'observations': [], 'actions': [], 'rewards': []}.
    :param model_path: path to a pretrained model for the MDP
    :param output_dir: directory to save the outputs of the run. The following directories will be created in the
    output directory: 'iterations', 'logs'
    :param device: device ('cpu' or 'cuda') to run the MOReL algorithm on"""

    # Significant portions of this function are based on the original MOReL implementation from:
    # https://github.com/aravindr93/mjrl/blob/v2/projects/morel/run_morel.py
    # We created our own implementation of the MOReL algorithm's MDP and USAD, however, we're using the same
    #   implementation of the planning algorithm that the original MOReL code uses.
    # To make that algorithm work, we had to replicate the setup from the original MOReL code.

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir + '/iterations'):
        os.mkdir(output_dir + '/iterations')
    if not os.path.exists(output_dir + '/logs'):
        os.mkdir(output_dir + '/logs')

    logger = DataLog()

    # create the environment using MJRL's gym environment class
    action_repeat = 2 # it seems like the original MOReL code uses action repeat of 2
    e = GymEnv(env_name, act_repeat=action_repeat)
    termination_function = getattr(e.env.env, "truncate_paths", None)

    # ===============================================================================
    # Setup the mdp, the policy, and the agent
    # ===============================================================================

    # if the model path is none, then the MDP will automatically train its dynamics model using the dataset
    #   otherwise, the MDP will load the dynamics model from the model path
    mdp = MDP(dataset=dataset, env_name=env_name, num_epochs=dynamics_model_training_epochs,
              negative_reward=negative_reward, device=device, model_path=model_path)
    if model_save_path is not None:
        mdp.save(model_save_path)

    policy = MLP(e.spec, hidden_sizes=npg_kwargs['policy_size'],
                        init_log_std=npg_kwargs['init_log_std'], min_log_std=npg_kwargs['min_log_std'])

    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=256, epochs=1,  learn_rate=1e-3,
                           device=device)

    # the agent will use the pessemistic MDP for its learned model
    agent = ModelBasedNPG(learned_model=[mdp], env=e, policy=policy, baseline=baseline,
                          normalized_step_size=npg_kwargs['step_size'], save_logs=True,
                          termination_function=termination_function, device=device,
                          **npg_kwargs['npg_hp'])

    # ===============================================================================
    # Create the initial state buffer and log some statistics about the dataset/MDP
    # ===============================================================================

    init_states_buffer = [trajectory['observations'][0] for trajectory in dataset]
    best_perf = -1e8

    s = np.concatenate([trajectory['observations'][:-1] for trajectory in dataset])
    a = np.concatenate([trajectory['actions'][:-1] for trajectory in dataset])
    sp = np.concatenate([trajectory['observations'][1:] for trajectory in dataset])
    rollout_score = np.mean([np.sum(trajectory['rewards']) for trajectory in dataset])
    num_samples = np.sum([len(trajectory['rewards']) for trajectory in dataset])

    logger.log_kv('fit_epochs', dynamics_model_training_epochs)
    logger.log_kv('rollout_score', rollout_score)
    logger.log_kv('iter_samples', num_samples)
    logger.log_kv('num_samples', num_samples)
    try:
        rollout_metric = e.env.env.evaluate_success(dataset)
        logger.log_kv('rollout_metric', rollout_metric)
    except:
        pass

    loss_general = mdp.compute_loss(s, a, sp)
    logger.log_kv('MDP Loss', loss_general)

    print("Model learning statistics")
    print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                               logger.get_current_log().items()))
    print(tabulate(print_data))
    logger.log_kv('act_repeat', action_repeat)


    # ===============================================================================
    # Policy Optimization Loop (for the Model-Based NPG algorithm)
    # ===============================================================================
    print('Starting policy optimization...')
    for npg_iteration in range(npg_kwargs['num_updates']):
        ts = timer.time()
        agent.to(device)

        print('sampling from initial state distribution')
        buffer_rand_idx = np.random.choice(len(init_states_buffer), size=npg_kwargs['num_gradient_trajectories'],
                                           replace=True).tolist()
        init_states = [init_states_buffer[idx] for idx in buffer_rand_idx]


        train_stats = agent.train_step(N=len(init_states), init_states=init_states, **npg_kwargs)
        logger.log_kv('train_score', train_stats[0])
        agent.policy.to('cpu')

        # evaluate true policy performance
        if eval_rollouts > 0:
            print("Performing validation rollouts ... ")
            # set the policy device back to CPU for env sampling
            eval_paths = evaluate_policy(agent.env, agent.policy, agent.learned_model[0], noise_level=0.0,
                                         real_step=True, num_episodes=eval_rollouts, visualize=False)
            eval_score = np.mean([np.sum(p['rewards']) for p in eval_paths])
            logger.log_kv('eval_score', eval_score)
            try:
                eval_metric = e.env.env.evaluate_success(eval_paths)
                logger.log_kv('eval_metric', eval_metric)
            except:
                pass
        else:
            eval_score = -1e8

        # track best performing policy
        policy_score = eval_score if eval_rollouts > 0 else rollout_score
        if policy_score > best_perf:
            best_policy = copy.deepcopy(policy)  # safe as policy network is clamped to CPU
            best_perf = policy_score

        tf = timer.time()
        logger.log_kv('iter_time', tf - ts)
        for key in agent.logger.log.keys():
            logger.log_kv(key, agent.logger.log[key][-1])
        print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                   logger.get_current_log_print().items()))
        print(tabulate(print_data))
        logger.save_log(output_dir + '/logs')

        if npg_iteration > 0 and npg_iteration % save_freq == 0:
            # convert to CPU before pickling
            agent.to('cpu')
            # make observation mask part of policy for easy deployment in environment
            old_in_scale = policy.in_scale
            for pi in [policy, best_policy]: pi.set_transformations(in_scale=1.0 / e.obs_mask)
            pickle.dump(agent, open(output_dir + f'/iterations/{env_name}_agent_{npg_iteration}.pickle', 'wb'))
            pickle.dump(policy, open(output_dir + f'/iterations/{env_name}_policy_{npg_iteration}.pickle', 'wb'))
            pickle.dump(best_policy, open(output_dir + f'/iterations/{env_name}_best_policy.pickle', 'wb'))
            agent.to(device)
            for pi in [policy, best_policy]: pi.set_transformations(in_scale=old_in_scale)
            make_train_plots(log=logger.log, keys=['rollout_score', 'eval_score', 'rollout_metric', 'eval_metric'],
                             x_scale=float(action_repeat), y_scale=1.0, save_loc=output_dir + '/logs/')

    # final save
    pickle.dump(agent, open(output_dir + '/iterations/agent_final.pickle', 'wb'))
    policy.set_transformations(in_scale=1.0 / e.obs_mask)
    pickle.dump(policy, open(output_dir + '/iterations/policy_final.pickle', 'wb'))

    # lastly, use mjrl's plotting function to make some plots
    # it uses argparser, not a function, so we need to call it from the terminal

    filename = 'mjrl/mjrl/utils/plot_from_logs.py'
    plot_file = os.path.join(output_dir, env_name + '.png')
    data_file = os.path.join(output_dir, 'logs', 'logs.pickle')
    os.system(f'python {filename} --data {data_file} --output {plot_file}')

if __name__ == '__main__':
    # these command line arguments are the hyperparamters and other settings that can change between environments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='<Required> environment to run on')
    parser.add_argument('--dataset', type=str, required=True, help='<Required> dataset to use for training')
    parser.add_argument('--device', type=str, default='cuda', help='device to run on')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help=f'<Default: {DEFAULT_MODEL_PATH}> pretrained MDP model to use for training. If none, the model will be trained from scratch.')
    parser.add_argument('--model-save-path', type=str, default=None, help=f'<Default: {None}> if the model path is None, the model will be trained and saved to this path.')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help=f'<Default: {DEFAULT_OUTPUT_DIR}> output directory for logs and models')

    parser.add_argument('--model-training-epochs', type=int, default=DEFAULT_TRAINING_EPOCHS,
                        help=f'<Default: {DEFAULT_TRAINING_EPOCHS}> number of MDP model training epochs (not needed if using a pretrained model')
    parser.add_argument('--negative-reward', type=float, default=DEFAULT_NEGATIVE_REWARD, help=f'<Default: {DEFAULT_NEGATIVE_REWARD}> negative reward for unknown states')
    parser.add_argument('--num-npg-updates', type=int, default=DEFAULT_NPG_UPDATES, help=f'<Default: {DEFAULT_NPG_UPDATES}> number of NPG updates')
    parser.add_argument('--num-gradient-trajectories', type=int, default=DEFAULT_NUM_GRADIENT_TRAJECTORIES,
                        help=f'<Default: {DEFAULT_NUM_GRADIENT_TRAJECTORIES,}> number of trajectories to sample for gradient updates')
    parser.add_argument('--horizon', type=int, default=DEFAULT_HORIZON, help=f'<Default: {DEFAULT_HORIZON}> horizon for NPG updates')
    parser.add_argument('--init-log-std', type=float, default=DEFAULT_INIT_LOG_STD, help=f'<Default: {DEFAULT_INIT_LOG_STD}> initial log std for NPG updates')
    parser.add_argument('--min_log_std', type=float, default=DEFAULT_MIN_LOG_STD, help=f'<Default: {DEFAULT_MIN_LOG_STD}> minimum log std for NPG updates')
    parser.add_argument('--cg-steps', type=int, default=DEFAULT_CG_STEPS, help=f'<Default: {DEFAULT_CG_STEPS}> number of CG steps')

    parser.add_argument('--eval-rollouts', type=int, default=DEFAULT_EVAL_ROLLOUTS, help=f'<Default: {DEFAULT_EVAL_ROLLOUTS}> number of rollouts to evaluate')
    parser.add_argument('--save-freq', type=int, default=DEFAULT_SAVE_FREQ, help=f'<Default: {DEFAULT_SAVE_FREQ}> number of iterations between saves')
    args = parser.parse_args()

    npg_kwargs = dict(policy_size=(32, 32),
                      init_log_std=args.init_log_std,
                      min_log_std=args.min_log_std,
                      step_size=.02,
                      num_updates=args.num_npg_updates,
                      horizon=args.horizon,
                      num_gradient_trajectories=args.num_gradient_trajectories,
                      npg_hp=dict(FIM_invert_args={'iters': args.cg_steps, 'damping': 1e-4}),
                      gamma=0.999, # gamma and gae_lambda are not hyperparameters in the paper,
                      gae_lambda=.97,) # but they can be found in https://github.com/aravindr93/mjrl/blob/v2/projects/morel/configs/hopper_v3_morel.txt, so I included them anyways

    with open(args.dataset, 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)
        print(f'Loaded dataset {args.dataset}')

    run_morel(env_name=args.env, dataset=dataset, model_path=args.model, model_save_path=args.model_save_path,
              output_dir=args.output_dir, device=args.device,
              dynamics_model_training_epochs=args.model_training_epochs,
              negative_reward=args.negative_reward,
              eval_rollouts=args.eval_rollouts,
              save_freq=args.save_freq,
                npg_kwargs=npg_kwargs)