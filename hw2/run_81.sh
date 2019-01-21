#!/bin/bash
rm -rf data/hc1_*
for b in 10000 30000 50000
do
    for lr in 0.005 0.01 0.02
    do
        python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $b -lr $lr -rtg --nn_baseline -nt --exp_name hc1_b-$b\_lr-$lr
    done
done
python plot.py data/hc1_* --save_name p81