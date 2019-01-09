# CS294-112 HW 1: Imitation Learning

## Usage

To generate expert data for training, run

```bash
bash get_data.sh
```

Please note that `run_expert.py` has been modified from the original version. See `run_expert.py` for details. Expert data would be saved in `expert_data/`.

To run all experiments and generate tables and figures for the report, run

```bash
bash run.sh
```

All results would be saved in `experiments/` and `report.md`.


## Results

I used a 3-hidden-layer fully connected neural network with 100 nodes at each hidden layer, ReLU non-linearity after each hidden layer, and L2 loss for all experiments in Section 2 and 3. Details on model architecture and training routines could be found in `model.py`. Hyperparameters could be found in `main.py`.

### Question 2.2

|Task|Mean return (BC)|STD (BC)|Mean return (expert)|STD (expert)|
|---|---|---|---|---|
|Hopper-v2|1860.27|444.31|3779.27|2.97|
|Ant-v2|4803.19|165.30|4788.46|102.76|
|HalfCheetah-v2|4208.68|86.97|4105.91|72.27|
|Humanoid-v2|874.53|328.83|10380.54|58.80|
|Reacher-v2|-5.77|1.99|-3.45|1.51|
|Walker2d-v2|3142.17|822.97|5505.50|46.97|

### Question 2.3

I investigated the effect of the number of training epochs.

<p float="left">
  <img src="./experiments/Hopper-v2/behavioral_cloning/Hopper-v2_behavioral_cloning.png" width="350"/>
  <img src="./experiments/Ant-v2/behavioral_cloning/Ant-v2_behavioral_cloning.png" width="350"/>
  <img src="./experiments/HalfCheetah-v2/behavioral_cloning/HalfCheetah-v2_behavioral_cloning.png" width="350"/>
  <img src="./experiments/Humanoid-v2/behavioral_cloning/Humanoid-v2_behavioral_cloning.png" width="350"/>
  <img src="./experiments/Reacher-v2/behavioral_cloning/Reacher-v2_behavioral_cloning.png" width="350"/>
  <img src="./experiments/Walker2d-v2/behavioral_cloning/Walker2d-v2_behavioral_cloning.png" width="350"/>
</p>

### Question 3.2

<p float="left">
  <img src="./experiments/Hopper-v2/Hopper-v2.png" width="350"/>
  <img src="./experiments/Ant-v2/Ant-v2.png" width="350"/>
  <img src="./experiments/HalfCheetah-v2/HalfCheetah-v2.png" width="350"/>
  <img src="./experiments/Humanoid-v2/Humanoid-v2.png" width="350"/>
  <img src="./experiments/Reacher-v2/Reacher-v2.png" width="350"/>
  <img src="./experiments/Walker2d-v2/Walker2d-v2.png" width="350"/>
</p>

### Question 4.1

I experimented with a different loss, the smooth L1 loss. I performed experiments with behavioral cloning.

<p float="left">
  <img src="./experiments/Hopper-v2_smooth-l1/Hopper-v2.png" width="350"/>
  <img src="./experiments/Ant-v2_smooth-l1/Ant-v2.png" width="350"/>
  <img src="./experiments/HalfCheetah-v2_smooth-l1/HalfCheetah-v2.png" width="350"/>
  <img src="./experiments/Humanoid-v2_smooth-l1/Humanoid-v2.png" width="350"/>
  <img src="./experiments/Reacher-v2_smooth-l1/Reacher-v2.png" width="350"/>
  <img src="./experiments/Walker2d-v2_smooth-l1/Walker2d-v2.png" width="350"/>
</p>


## Original README

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.
