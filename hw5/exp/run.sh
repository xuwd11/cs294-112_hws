#rm -rf data/ac_PM_bc0_s8_PointMass-v0 data/ac_PM_hist_bc0.01_s8_PointMass-v0
#python train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model none -s 8 --exp_name PM_bc0_s8
#python train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model hist -bc 0.01 -s 8 --exp_name PM_hist_bc0.01_s8
#python plot.py data/ac_PM_bc0_s8_PointMass-v0 data/ac_PM_hist_bc0.01_s8_PointMass-v0 --save_name p1

#rm -rf data/ac_PM_rbf_bc0.01_s8_sig0.2_PointMass-v0
#python train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model rbf -bc 0.01 -s 8 -sig 0.2 --exp_name PM_rbf_bc0.01_s8_sig0.2
#python plot.py data/ac_PM_bc0_s8_PointMass-v0 data/ac_PM_rbf_bc0.01_s8_sig0.2_PointMass-v0 --save_name p2