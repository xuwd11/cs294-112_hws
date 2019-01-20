# CS294-112 HW 2: Policy Gradient

## Results
### Problem 1
#### 1a
For each term in equation 12, we have

<p align="center"><img src="/hw2/tex/344d33eb616b74dbcae4e4d2aae62c94.svg?invert_in_darkmode&sanitize=true" align=middle width=774.5332006499999pt height=280.5902187pt/></p>

Therefore, 

<p align="center"><img src="/hw2/tex/c27101b2bb742f3c10dfd4f7b27c299b.svg?invert_in_darkmode&sanitize=true" align=middle width=282.41141775pt height=47.60747145pt/></p>

#### 1b
**a** Future states and actions are independent of previous states and actions given the current state according to the Markov property of MDP.

**b**

<p align="center"><img src="/hw2/tex/6032ead988551c349c799b3f615ec60a.svg?invert_in_darkmode&sanitize=true" align=middle width=781.5363831pt height=320.773266pt/></p>

Therefore, 

<p align="center"><img src="/hw2/tex/c27101b2bb742f3c10dfd4f7b27c299b.svg?invert_in_darkmode&sanitize=true" align=middle width=282.41141775pt height=47.60747145pt/></p>

### Problem 4

<p float="left">
  <img src="./results/p4_sb.png" width="350"/>
  <img src="./results/p4_lb.png" width="350"/>
</p>

* Reward-to-go has better performance than the trajectory-centric one without advantage-centering; reward-to-go converges faster and has lower variance.
* Advantage centering helps reduce the variance after convergence.
* Larger batch size helps reduce the variance.

### Problem 5

<p float="left">
  <img src="./results/p5_ip_b-1000_lr-1e-2.png" width="350"/>
</p>


## Original README

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only file that you need to look at is `train_pg_f18.py`, which you will implement.

See the [HW2 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf) for further instructions.
