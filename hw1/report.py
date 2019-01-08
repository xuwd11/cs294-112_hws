import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATA_DIR = os.path.join('.', 'expert_data')
EXPERIMENTS_DIR = os.path.join('.', 'experiments')

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def plot_32(env_name):
    expert_returns = load_json(os.join.path(DATA_DIR, env_name + '.json'))