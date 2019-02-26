# CS294-112 HW 5c: Meta-Learning
## Usage

To run all experiments and plot figures for the report, run

```bash
bash run_1.sh
bash run_21.sh
bash run_22.sh
bash run_23.sh
bash run_24.sh
bash run_25.sh
bash run_31.sh
bash run_32.sh
bash run_33.sh
```

## Results
### 5.1 Problem 1
<p float="left">
  <img src="./results/p1.png" width="350"/>
</p>

### 5.2 Problem 2
<p float="left">
  <img src="./results/p2_mlp.png" width="350"/>
  <img src="./results/p2_gru.png" width="350"/>
</p>
<p float="left">
  <img src="./results/p2_1.png" width="350"/>
  <img src="./results/p2_15.png" width="350"/>
  <img src="./results/p2_30.png" width="350"/>
  <img src="./results/p2_45.png" width="350"/>
  <img src="./results/p2_60.png" width="350"/>
</p>
The recurrent architectures outperform the feed-forward architectures in all cases.

## Original README
Dependencies:
 * Python **3.5**
 * Numpy version 1.14.5
 * TensorFlow version 1.10.5
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==2.3.2

See the [HW5c PDF](hw5c_instructions.pdf) for further instructions.
