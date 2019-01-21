#!/bin/bash
rm -rf data/hc2_*
b=50000
lr=0.02
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $b -lr $lr -nt --exp_name hc2_b-$b\_lr-$lr --gpu 1
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $b -lr $lr -rtg -nt --exp_name hc2_b-$b\_lr-$lr\_rtg --gpu 1
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $b -lr $lr --nn_baseline -nt --exp_name hc2_b-$b\_lr-$lr\_nn_baseline --gpu 1
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $b -lr $lr -rtg --nn_baseline -nt --exp_name hc2_b-$b\_lr-$lr\_rtg_nn_baseline --gpu 1
python plot.py data/hc2_* --save_name p82