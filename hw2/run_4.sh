#!/bin/bash
rm -rf data/sb_* data/lb_*
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna -nt --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna -nt --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -nt --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna -nt --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -nt -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -nt --exp_name lb_rtg_na
python plot.py data/sb_no_rtg_dna_CartPole-v0 data/sb_rtg_dna_CartPole-v0 data/sb_rtg_na_CartPole-v0 --save_name p4_sb
python plot.py data/lb_no_rtg_dna_CartPole-v0 data/lb_rtg_dna_CartPole-v0 data/lb_rtg_na_CartPole-v0 --save_name p4_lb