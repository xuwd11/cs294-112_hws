#!/bin/bash
rm -rf data/hc11_*
for b in 10000
do
    for lr in 0.005 0.01 0.02
    do
        python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $b -lr $lr -rtg --nn_baseline -nt --exp_name hc11_b-$b\_lr-$lr
    done
done
python plot.py data/hc11_* --save_name p811