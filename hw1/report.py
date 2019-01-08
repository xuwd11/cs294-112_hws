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

def table_21(env_list):
    pass

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

def get_plots_32(env_list):
    with open('report.md', 'a') as f:
        f.write('### Question 3.2\n')
    for env_name in env_list:
        path = plot_32(env_name)
        with open('report.md', 'a') as f:
            f.write('<img src="{}" width="200"/>'.format(path))

if __name__ == '__main__':
    env_list = ['Hopper-v2']
    get_plots_32(env_list)