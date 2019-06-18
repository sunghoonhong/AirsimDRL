import os
import csv
import numpy as np
import argparse
from matplotlib import pyplot as plt


def smooth(arr, n):
    end = -(len(arr)%n)
    if end == 0:
      end = None
    arr = np.reshape(arr[:end], (-1, n))
    arr = np.mean(arr, axis=1)
    return arr

def drawall(name, x, metrics, n=100, begin=0):
    x = smooth(x[-begin:], n)
    for i, metric in enumerate(metrics):
        metrics[i] = smooth(metric[-begin:], n)

    plt.plot(x, metrics[0], label=name, linewidth=3)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--begin', type=int, default=0)
    args = parser.parse_args()

    dir ='save_graph/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    plt.figure(figsize=(15,5))
    plt.xlabel('Episode')
    plt.ylabel('Best record')
    plt.title('Result')
    for name in ['ra2c', 'rdqn']:#, 'ppo', 'rddpg_per', 'td3_per']:
        filename = './save_stat/' + name + '_stat.csv'
    
        bestY = []

        with open(filename, 'r') as f:
            read = csv.reader(f)
            for i, row in enumerate(read):
                if 'ppo' in name:
                    bestY.append(float(row[0]))
                else:
                    bestY.append(float(row[3]))
            episodes = [i for i in range(len(bestY))]
            metrics = [
                bestY
            ]
        
        drawall(name, episodes, metrics, args.n, args.begin)

        
    plt.legend()
    plt.savefig(dir + '/result.png')