#!/usr/bin/env bash

python plot.py data/pm_gru_h-60_gs-1 data/pm_gru_h-60_gs-5 data/pm_gru_h-60_gs-10 --legend gru_h-60_gs-1_train gru_h-60_gs-5_train gru_h-60_gs-10_train --save_name p3_train
python plot.py data/pm_gru_h-60_gs-1 data/pm_gru_h-60_gs-5 data/pm_gru_h-60_gs-10 --value ValAverageReturn --legend gru_h-60_gs-1_test gru_h-60_gs-5_test gru_h-60_gs-10_test --save_name p3_test

python plot_3.py data/pm_gru_h-60_gs-1 --save_name p3_1
python plot_3.py data/pm_gru_h-60_gs-5 --save_name p3_5
python plot_3.py data/pm_gru_h-60_gs-10 --save_name p3_10