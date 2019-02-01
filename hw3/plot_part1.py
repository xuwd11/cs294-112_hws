import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def read_result(exp_name):
    path = os.path.join('data', exp_name, 'log.txt')
    return pd.read_csv(path, sep='\t')

def add_plot(data, var_name, label=''):
    sns.set(style="darkgrid", font_scale=1.5)
    plt.plot(data['Timestep'], data[var_name], \
             label=label, alpha=0.8)

def plot_11(data):
    r1, r2, r3, r4 = data
    plt.figure()
    add_plot(r1, 'MeanReward100Episodes', 'MeanReward100Episodes');
    add_plot(r1, 'BestMeanReward', 'BestMeanReward');
    plt.xlabel('Time step');
    plt.ylabel('Reward');
    plt.legend();
    plt.savefig(
        os.path.join('results', 'p11.png'),
        bbox_inches='tight',
        transparent=True,
        pad_inches=0.1
    )
    
def plot_12(data):
    r1, r2, r3, r4 = data
    plt.figure()
    add_plot(r1, 'MeanReward100Episodes');
    add_plot(r1, 'BestMeanReward', 'vanilla DQN');
    add_plot(r2, 'MeanReward100Episodes');
    add_plot(r2, 'BestMeanReward', 'double DQN');
    plt.xlabel('Time step');
    plt.ylabel('Reward');
    plt.legend();
    plt.savefig(
        os.path.join('results', 'p12.png'),
        bbox_inches='tight',
        transparent=True,
        pad_inches=0.1
    )
    
def plot_13(data):
    r1, r2, r3, r4 = data
    plt.figure()
    add_plot(r3, 'MeanReward100Episodes');
    add_plot(r3, 'BestMeanReward', 'gamma = 0.9');
    add_plot(r2, 'MeanReward100Episodes');
    add_plot(r2, 'BestMeanReward', 'gamma = 0.99');
    add_plot(r4, 'MeanReward100Episodes');
    add_plot(r4, 'BestMeanReward', 'gamma = 0.999');
    plt.legend();
    plt.xlabel('Time step');
    plt.ylabel('Reward');
    plt.savefig(
        os.path.join('results', 'p13.png'),
        bbox_inches='tight',
        transparent=True,
        pad_inches=0.1
    )
    
def main():
    if not os.path.exists('results'):
        os.makedirs('results')
    r1 = read_result('PongNoFrameskip-v4_bq')
    r2 = read_result('PongNoFrameskip-v4_dq')
    r3 = read_result('PongNoFrameskip-v4_dq_gamma-0_9')
    r4 = read_result('PongNoFrameskip-v4_dq_gamma-0_999')
    data = (r1, r2, r3, r4)
    plot_11(data)
    plot_12(data)
    plot_13(data)
    
if __name__ == '__main__':
    main()