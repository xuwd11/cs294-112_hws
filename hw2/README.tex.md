# CS294-112 HW 2: Policy Gradient

## Results
### Problem 1
#### 1a
For each term in equation 12, we have

\begin{align*}
\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log \pi_\theta(a_t|s_t)(b(s_t))]
&=\int p_\theta(\tau)\nabla_\theta \log \pi_\theta(a_t|s_t)(b(s_t))\text{d}\tau\\
&=\iiint p_\theta(s_t,a_t) p_\theta(\tau/s_t,a_t|s_t,a_t)b(s_t)\nabla_\theta \log \pi_\theta(a_t|s_t)\text{d}(\tau/s_t,a_t)\text{d}a_t\text{d}s_t\\
&=\iint p_\theta(s_t,a_t) b(s_t)\nabla_\theta \log \pi_\theta(a_t|s_t)\left(\int p_\theta(\tau/s_t,a_t|s_t,a_t)\text{d}(\tau/s_t,a_t)\right)\text{d}a_t\text{d}s_t\\
&=\iint \frac{p_\theta(s_t,a_t)}{\pi_\theta(a_t|s_t)} b(s_t)\nabla_\theta \pi_\theta(a_t|s_t)\text{d}a_t\text{d}s_t\\
&=\int p_\theta(s_t)b(s_t)\left(\nabla_\theta\int\pi_\theta(a_t|s_t)\text{d}a_t\right)\text{d}s_t\\
&=\int p_\theta(s_t)b(s_t)(\nabla_\theta 1)\text{d}s_t\\
&=0
\end{align*}

Therefore, 

$$\sum_{t=1}^{T}\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log \pi_\theta(a_t|s_t)(b(s_t))] = 0$$

#### 1b
**a** Future states and actions are independent of previous states and actions given the current state according to the Markov property of MDP.

**b**

\begin{align*}
\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log \pi_\theta(a_t|s_t)(b(s_t))]
&=\int p_\theta(\tau)\nabla_\theta \log \pi_\theta(a_t|s_t)(b(s_t))\text{d}\tau\\
&=\iint p_\theta(s_{1:t},a_{1:t-1}) p_\theta(s_{t+1:T},a_{t:T}|s_{1:t},a_{1:t-1})b(s_t)\nabla_\theta \log \pi_\theta(a_t|s_t)\text{d}s_{1:T}\text{d}a_{1:T}\\
&=\iint p_\theta(s_{1:t},a_{1:t-1}) p_\theta(s_{t+1:T},a_{t:T}|s_t)b(s_t)\nabla_\theta \log \pi_\theta(a_t|s_t)\text{d}s_{1:T}\text{d}a_{1:T}\\
&=\iint p_\theta(s_t) p_\theta(s_{t+1:T},a_{t:T}|s_t)b(s_t)\nabla_\theta \log \pi_\theta(a_t|s_t)\text{d}s_{t:T}\text{d}a_{t:T}\\
&=\iint \frac{p_\theta(s_t) p_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)} b(s_t)\nabla_\theta \pi_\theta(a_t|s_t)\text{d}a_t\text{d}s_t\\
&=\int p_\theta(s_t)b(s_t)\left(\nabla_\theta\int\pi_\theta(a_t|s_t)\text{d}a_t\right)\text{d}s_t\\
&=\int p_\theta(s_t)b(s_t)(\nabla_\theta 1)\text{d}s_t\\
&=0
\end{align*}

Therefore, 

$$\sum_{t=1}^{T}\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log \pi_\theta(a_t|s_t)(b(s_t))] = 0$$

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
