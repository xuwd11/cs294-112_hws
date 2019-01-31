#!/bin/bash
rm -rf data/PongNoFrameskip-v4_bq
python run_dqn_atari.py PongNoFrameskip-v4_bq --gpu 0