#!/bin/bash
b=1000
lr=1e-3
rm -rf data/ip93_*
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $b -lr $lr -ps 1 -rtg -nt --exp_name ip93_b-$b\_lr-$lr\_pg-step-1
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $b -lr $lr -ps 5 -rtg -nt --exp_name ip93_b-$b\_lr-$lr\_pg-step-5
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $b -lr $lr -ps 20 -rtg -nt --exp_name ip93_b-$b\_lr-$lr\_pg-step-20
python plot.py data/ip93_* --save_name p93