#!/bin/bash
rm -rf data/hc13_*
for b in 50000
do
    for lr in 0.005 0.01 0.02
    do
        python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $b -lr $lr -rtg --nn_baseline -nt --exp_name hc13_b-$b\_lr-$lr
    done
done
python plot.py data/hc13_* --save_name p813