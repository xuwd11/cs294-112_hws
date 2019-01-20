#!/bin/bash
b=1000
lr=1e-2
rm -rf data/ip_b-$b\_lr-$lr\_InvertedPendulum-v2
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $b -lr $lr -rtg -nt --exp_name ip_b-$b\_lr-$lr
python plot.py data/ip_b-$b\_lr-$lr\_InvertedPendulum-v2 --save_name p5_ip_b-$b\_lr-$lr