# CS294-112 HW 2: Policy Gradient

### Problem 1
#### 1a
For each term in equation 12, we have

<p align="center"><img src="/hw2/tex/344d33eb616b74dbcae4e4d2aae62c94.svg?invert_in_darkmode&sanitize=true" align=middle width=774.5332006499999pt height=280.5902187pt/></p>

Therefore, 

<p align="center"><img src="/hw2/tex/c27101b2bb742f3c10dfd4f7b27c299b.svg?invert_in_darkmode&sanitize=true" align=middle width=282.41141775pt height=47.60747145pt/></p>

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
