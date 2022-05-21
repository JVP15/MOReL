# MOReL
This is an implementation of the MOReL offline reinforcement learning algorithm using Python. The full paper can be found here: [https://arxiv.org/pdf/2005.05951.pdf](https://arxiv.org/pdf/2005.05951.pdf)

## Installation

This project relies on the [MJRL](https://github.com/aravindr93/mjrl/tree/v2) repo. Clone that repository into this folder using:

`git clone https://github.com/aravindr93/mjrl.git`

then navigate into the MJRL repo with `cd mjrl` and checkout the branch `v2` (which this project relies on) using `git checkout v2`. You will then need to follow [these steps](https://github.com/aravindr93/mjrl/tree/master/setup#installation) to properly install MJRL. 

This implementation also relies on [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) and [Stable Baselines3-Contrib](https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html), which you can install with:

`pip install stable-baselines3[extra]`
`pip install sb3-contrib`

## Training Collection Policies

We have pre-trained collection policies (TRPO trained until it reached a score of 1000) for Ant-v2, Hopper-v2, and Walker2d-v2, which can be found in the `pretrained_policies` folder. To train your own collection policy, you can use `train_collection_policy.py`. Run `train_collection_policy.py --help` to see all available options. Here is the command we used to train the Hopper-v2 policy:

`python train_collection_policy.py --env Hopper-v2 --target-reward 1000 --policy-dir trained_policies

## Collecting Datasets

Datasets take up a lot of space, and so we only have a dataset that contains 10,000 timesteps from Ant-v2 collected using the pure collection policy. You can create your own datasets using `collect_dataset.py`. Run `collect_dataset.py --help` to see all available options. Here is how we collected the Ant-v2 dataset:

`python collect_dataset.py --env Ant-v2 --policy trained_policies/TRPO_Ant-v2_1000.zip --dir ./dataset`

To collect the same dataset with gaussian noise with a std of 0.1, run:

`python collect_dataset.py --env Ant-v2 --policy trained_policies/TRPO_Ant-v2_1000.zip --dir ./dataset --action-noise gauss 0.1`

To collect the same dataset with partially random actions instead of noise, run:

`python collect_dataset.py --env Ant-v2 --policy trained_policies/TRPO_Ant-v2_1000.zip --dir ./dataset --action-noise random 0.1`

Here is how to collect 1,000,000 timesteps for the Hopper-v2 environment using the pure collection policy. Note the trajectory length has been reduced to 400 to follow the dataset collection methods performed in the paper:

`python collect_dataset.py --env Hopper-v2 --policy trained_policies/TRPO_Hopper-v2_1000.zip --dir ./dataset --iterations 1000000 --trajectory-length 400` 

## Training Models

Training dynamics models on datasets with 1,000,000 timesteps takes hours. So we have provided a pretrained MDP dynamics model and USAD dynamics models (trained on 1,000,000 pure policy Hopper-v2 dataset) in `trained_models`. To train your own models, use `mdp.py`. Here is an example to train an MDP and the USAD on the Ant-v2 dataset:

`python mdp.py --dataset dataset/TRPO_Ant-v2_10000 --env Ant-v2 --output trained_models/MDP_Ant-v2_10000 --usad-output trained_models/USAD_Ant-v2_10000 --epochs 10`

Here is the command to train an MDP and USAD on the 1,000,000 step pure policy Hopper-v2 dataset (to replicate the results from the paper):

`python mdp.py --dataset dataset/TRPO_Hopper-v2_1000000 --env Hopper-v2 --output trained_models/MDP_Hopper-v2_1e6 --negative-reward 50 --usad-output trained_models/USAD_Hopper-v2_1e6`

## Running MOReL 

Once you have a dataset and (optionally) a pretrained MDP and USAD, you can run the MOReL algorithm using `morel.py`. Run `morel.py` to see all of the options and hyperparameters available for tuning. To run MOReL on Ant-v2 using the dataset with 10,000 timesteps and the hyperparameters for Ant-v2 from the paper, run:

`python morel.py --env Ant-v2 --dataset dataset/TRPO_Ant-v2_10000 --output-dir ant_output --model trained_models/MDP_Ant-v2_10000 --usad-model trained_models/USAD_Ant-v2_10000`

To recreate the results from the paper for Hopper-v2 trained on the pure policy dataset of 1,000,000 timesteps, run: 

`python morel.py --env Hopper-v2 --dataset dataset/TRPO_Hopper-v2_1000000 --model trained_models/MDP_Hopper-v2_1e6 --usad-model trained_models/USAD_Hopper-v2_1e6 --output-dir hopper_output_naive --horizon 400 --negative-reward 50 --num-npg-updates 500 --init-log-std -.25 --num-gradient-trajectories 50 --cg-steps 25`

