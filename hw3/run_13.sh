#!/bin/bash
rm -rf data/PongNoFrameskip-v4_dq_gamma-0_9
python run_dqn_atari.py PongNoFrameskip-v4_dq_gamma-0_9 --double_q --gamma 0.9 --gpu 1