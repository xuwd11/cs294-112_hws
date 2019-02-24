#!/usr/bin/env bash

python plot.py data/pm_mlp_h-1_gs-1 data/pm_mlp_h-15_gs-1 data/pm_mlp_h-30_gs-1 data/pm_mlp_h-45_gs-1 data/pm_mlp_h-60_gs-1 --save_name p2_mlp
python plot.py data/pm_gru_h-1_gs-1 data/pm_gru_h-15_gs-1 data/pm_gru_h-30_gs-1 data/pm_gru_h-45_gs-1 data/pm_gru_h-60_gs-1 --save_name p2_gru
python plot.py data/pm_mlp_h-1_gs-1 data/pm_gru_h-1_gs-1 --save_name p2_1
python plot.py data/pm_mlp_h-15_gs-1 data/pm_gru_h-15_gs-1 --save_name p2_15
python plot.py data/pm_mlp_h-30_gs-1 data/pm_gru_h-30_gs-1 --save_name p2_30
python plot.py data/pm_mlp_h-45_gs-1 data/pm_gru_h-45_gs-1 --save_name p2_45
python plot.py data/pm_mlp_h-60_gs-1 data/pm_gru_h-60_gs-1 --save_name p2_60