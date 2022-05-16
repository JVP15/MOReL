import numpy as np

def ant_reward(s,a):
    # mimics reward function for the ant-v2 environment from:
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py (for values)
    # and
    # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py
    # for information about what each value in the state represents
    # this isn't the exact same reward function as the one used by Ant-v2, but it is pretty accurate
    # expects s and a to be a batche of states and actions

    healthy_reward = 1.0
    x_velocity = s[:, 13] # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py#L69
    forward_reward = x_velocity

    rewards = healthy_reward + forward_reward

    contact_force = s[:, 27:] # https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py#L85
    contact_cost_weight = .5 * 1e-3
    contact_cost = contact_cost_weight * np.sum(np.square(np.clip(contact_force, -1, 1)), axis=-1)
    control_cost_weight = .5
    control_cost = control_cost_weight * np.sum(np.square(a), axis=-1)

    costs = contact_cost + control_cost
    # this reward function is typically off by about .0045, so we can make the reward function more accurate by subtracting this amount
    total_rewards = rewards - costs - .0045

    return total_rewards

def ant_termination_function(trajectories):
    # this function truncates a trajectory if the ant reaches a termination state before the trajectory is done
    # healthy algorithm taken from
    # https://github.com/openai/gym/blob/bf688c3efef49b557a11280f92e98154e7b7c264/gym/envs/mujoco/ant_v3.py#L230
    # interface is taken from
    # https://github.com/aravindr93/mjrl/blob/v2/projects/model_based_npg/utils/reward_functions/gym_hopper.py
    min_z = .2
    max_z = 1.0

    for trajectory in trajectories:
        states = trajectory['observations']
        z_torso = states[:, 0]

        T = states.shape[0]
        t = 0
        done = False
        while t < T and done is False:
            done = not (np.isinfinite(states[t]).all() and min_z <= z_torso[t] <= max_z)
            t = t + 1
            T = t if done else T
        trajectory["observations"] = trajectory["observations"][:T]
        trajectory["actions"] = trajectory["actions"][:T]
        trajectory["rewards"] = trajectory["rewards"][:T]
        trajectory["terminated"] = done

    return trajectories

# observaion mask for scaling. Without scaling, the planning algorithm seems to perform poorly
# 1.0 for positions and dt=0.02 for velocities. Taken from
# https://github.com/aravindr93/mjrl/blob/v2/projects/model_based_npg/utils/reward_functions/gym_hopper.py
hopper_obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

def hopper_reward(s, a):
    # mimics reward function for the hopper-v2 environment from:
    # https://github.com/aravindr93/mjrl/blob/v2/projects/model_based_npg/utils/reward_functions/gym_hopper.py
    # expects s and a to be a batche of states and actions

    obs = np.clip(s, -10.0, 10.0)
    act = np.clip(a, -1.0, 1.0)
    vel_x = obs[:, -6] / 0.02
    power = np.sum(np.square(act), axis=-1)
    height = obs[:, 0]
    ang = obs[:,1]
    alive_bonus = 1.0 * (height > 0.7) * (np.abs(ang) <= 0.2)
    rewards = vel_x + alive_bonus - 1e-3 * power
    return rewards

def hopper_termination_function(paths):
    # taken from
    # https://github.com/aravindr93/mjrl/blob/v2/projects/model_based_npg/utils/reward_functions/gym_hopper.py
    for path in paths:
        obs = path["observations"]
        height = obs[:, 0]
        angle = obs[:, 1]
        T = obs.shape[0]
        t = 0
        done = False
        while t < T and done is False:
            done = not ((np.abs(obs[t]) < 10).all() and (height[t] > 0.7) and (np.abs(angle[t]) < 0.15))
            t = t + 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths
