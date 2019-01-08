import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DATA_DIR = os.path.join('.', 'expert_data')
EXPERIMENTS_DIR = os.path.join('.', 'experiments')

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_table_22(env_list):
    with open('report.md', 'w') as f:
        f.write('### Question 2.2\n\n')
        f.write('|Task|Mean return (BC)|STD (BC)|Mean return (expert)|STD (expert)|\n')
        f.write('|---|---|---|---|---|\n')
        for env_name in env_list:
            expert = load_json(os.path.join(DATA_DIR, env_name + '.json'))
            exp_dir = os.path.join(EXPERIMENTS_DIR, env_name)
            bc = load_json(os.path.join(exp_dir, 'behavioral_cloning', env_name + '_results.json'))
            f.write(
                '|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|\n'.format(
                    env_name,
                    bc['best_return'],
                    bc['best_return_std'],
                    expert['mean_return'],
                    expert['std_return']
                )
            )
        f.write('\n')

def plot_32(env_name):
    expert = load_json(os.path.join(DATA_DIR, env_name + '.json'))
    exp_dir = os.path.join(EXPERIMENTS_DIR, env_name)
    bc = load_json(os.path.join(exp_dir, 'behavioral_cloning', env_name + '_results.json'))
    dagger = load_json(os.path.join(exp_dir, 'dagger', env_name + '_results.json'))
    
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.plot(
        np.arange(1, len(bc['epochs']) + 1),
        np.mean(bc['returns'], axis=-1),
        color='b',
        label='Behavioral Cloning'
    )
    plt.errorbar(
        np.arange(1, len(bc['epochs']) + 1),
        np.mean(bc['returns'], axis=-1),
        np.std(bc['returns'], axis=-1),
        fmt='.',
        color='b'
    )
    
    plt.plot(
        np.arange(1, len(dagger['epochs']) + 1),
        np.mean(dagger['returns'], axis=-1),
        color='r',
        label='DAgger'
    )
    plt.errorbar(
        np.arange(1, len(dagger['epochs']) + 1),
        np.mean(dagger['returns'], axis=-1),
        np.std(dagger['returns'], axis=-1),
        fmt='.',
        color='r'
    )
    
    plt.fill_between(
        np.arange(1, len(bc['epochs']) + 1),
        expert['mean_return'] - expert['std_return'],
        expert['mean_return'] + expert['std_return'],
        label='Expert',
        color='g'
    )
    plt.xlabel('DAgger iterations');
    plt.ylabel('Return');
    plt.legend(loc='best');
    plt.title(env_name);
    plt.savefig(
        os.path.join(exp_dir, env_name + '.png'),
        bbox_inches='tight',
        transparent=True,
        pad_inches=0.1
    )
    return os.path.join(exp_dir, env_name + '.png')

def plot_41(env_name):
    expert = load_json(os.path.join(DATA_DIR, env_name + '.json'))
    bc = load_json(os.path.join(EXPERIMENTS_DIR, env_name, 'behavioral_cloning', env_name + '_results.json'))
    bc1 = load_json(os.path.join(
        EXPERIMENTS_DIR, 
        env_name + '_smooth-l1', 
        'behavioral_cloning', 
        env_name + '_smooth-l1_results.json'))
    
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.plot(
        bc['epochs'],
        np.mean(bc['returns'], axis=-1),
        color='b',
        label='BC (l2 loss)'
    )
    plt.errorbar(
        bc['epochs'],
        np.mean(bc['returns'], axis=-1),
        np.std(bc['returns'], axis=-1),
        fmt='.',
        color='b'
    )
    
    plt.plot(
        bc1['epochs'],
        np.mean(bc1['returns'], axis=-1),
        color='r',
        label='BC (smooth l1 loss)'
    )
    plt.errorbar(
        bc1['epochs'],
        np.mean(bc1['returns'], axis=-1),
        np.std(bc1['returns'], axis=-1),
        fmt='.',
        color='r'
    )
    
    plt.fill_between(
        bc1['epochs'],
        expert['mean_return'] - expert['std_return'],
        expert['mean_return'] + expert['std_return'],
        label='Expert',
        color='g'
    )
    plt.xlabel('DAgger iterations');
    plt.ylabel('Return');
    plt.legend(loc='best');
    plt.title(env_name);
    plt.savefig(
        os.path.join(EXPERIMENTS_DIR, env_name + '_smooth-l1', env_name + '.png'),
        bbox_inches='tight',
        transparent=True,
        pad_inches=0.1
    )
    return os.path.join(EXPERIMENTS_DIR, env_name + '_smooth-l1', env_name + '.png')

def get_plots_23(env_list):
    with open('report.md', 'a') as f:
        f.write('### Question 2.3\n\n<p float="left">\n')
        for env_name in env_list:
            f.write('  <img src="{}" width="350"/>\n'.format(
                os.path.join(
                    EXPERIMENTS_DIR, 
                    env_name, 
                    'behavioral_cloning', 
                    env_name + '_behavioral_cloning.png'
                )
            ))
        f.write('</p>\n\n')

def get_plots_32(env_list):
    with open('report.md', 'a') as f:
        f.write('### Question 3.2\n\n<p float="left">\n')
        for env_name in env_list:
            path = plot_32(env_name)
            f.write('  <img src="{}" width="350"/>\n'.format(path))
        f.write('</p>\n\n')

def get_plots_41(env_list):
    with open('report.md', 'a') as f:
        f.write('### Question 4.1\n\n<p float="left">\n')
        for env_name in env_list:
            path = plot_41(env_name)
            f.write('  <img src="{}" width="350"/>\n'.format(path))
        f.write('</p>\n\n')

if __name__ == '__main__':
    env_list = ['Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2']
    get_table_22(env_list)
    get_plots_23(env_list)
    get_plots_32(env_list)
    get_plots_41(env_list)