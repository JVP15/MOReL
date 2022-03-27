# MOReL
This is an implementation of the MOReL offline reinforcement learning algorithm using Python and Tensorflow. The full paper can be found here: [https://arxiv.org/pdf/2005.05951.pdf](https://arxiv.org/pdf/2005.05951.pdf)

# MOReL
This is an implementation of the MOReL offline reinforcement learning algorithm using Python and Tensorflow. The full paper can be found here: [https://arxiv.org/pdf/2005.05951.pdf](https://arxiv.org/pdf/2005.05951.pdf)

# To-Do List

## :x: Dynamic Model Learning 
MOReL uses a Gaussian dynamics `P(*|s,a) = N(f(s,a), Σ)`. 

* :x: Train the neural network used by `f(s,a)` on the dataset using maximum likelihood estimation
* :x: Implement `f(s,a)` (see the bottom of [page 6](https://arxiv.org/pdf/2005.05951.pdf#page=6))

## :x: USAD
MOReL uses an Unknown State-Action Detector (USAD) to identify what states are known and unknown. This is implemented by creating an ensemble of models trained using different minibatches of the dataset. 

* :x: Implement `disc(s,a)` (see top of [page 7](https://arxiv.org/pdf/2005.05951.pdf#page=7))
* :x: Implement USAD (for threshold, see section C.5 on [page 21](https://arxiv.org/pdf/2005.05951.pdf#page=21))

## :x: Dataset

The MOReL algorithm was trained and tested using 4 environments: HalfCheetah-v2, Hopper-v2, Walker2D-v2, and ~~Headcrab~~ Ant-v2. 5 datasets were generated for each environment (see section 5 on [page 7](https://arxiv.org/pdf/2005.05951.pdf#page=7) for a breakdown of how they were created). Each dataset contains 1,000,000 timesteps. The paper used a partially trained policy π to generate the dataset. This policy was trained using a model-based natural policy gradeint (NPG) algorithm from [https://sites.google.com/view/mbrl-game](https://sites.google.com/view/mbrl-game)

* :x: Implement the policy training algorithm (this is the same as the task in the Planning section)
* :x: Train the collection policy (see see section 5 on [page 7](https://arxiv.org/pdf/2005.05951.pdf#page=7) for a breakdown on how the policy was trained)
* :x: Create the datasets (due to time and resource constraints, we may not recreate all of the datasets from the paper)

## :x: Planning

The policy used in the MOReL paper plans using the dynamics model and the USAD. It is trained using a model-based NPG algorithm from [https://sites.google.com/view/mbrl-game](https://sites.google.com/view/mbrl-game).

* :x: Implement the policy training algorithm (this is the same as the task in the Dataset section)
* :x: Train the policy 
