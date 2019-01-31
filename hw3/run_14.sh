#!/bin/bash
rm -rf data/PongNoFrameskip-v4_dq_gamma-0_999
python run_dqn_atari.py PongNoFrameskip-v4_dq_gamma-0_999 --double_q --gamma 0.999 --gpu 1