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

def drawall(name, x, metric, marker, n=100, begin=0):
    x = smooth(x[-begin:], n)
    # for i, metric in enumerate(metrics):
    metric = smooth(metric[-begin:], n)

    plt.plot(x, metric, label=name, marker=marker, markersize=10., linewidth=3)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--begin', type=int, default=0)
    args = parser.parse_args()

    dir ='save_graph/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    ylabels = [
        'Best Record', 'Get Goal Prob.', 'Step', 'Step per Record', 'Step for Goal' 
    ]
    linestyles = [
        '--', '-', '-.', ':', '-', '-'
    ]
    markers = [
        'o', '*', '.', 'd', 's', '>'
    ]
    agents = ['rddpg', 'td3_per', 'ppo', 'rddpg_per', 'ra2c', 'rdqn']

    for idx, ylabel in enumerate(ylabels):
        plt.figure(figsize=(15,5))
        # plt.xlabel('Episode', fontsize='large')
        plt.ylabel(ylabel, fontsize='large')
        plt.title(ylabel, fontsize='x-large')
        for ix, name in enumerate(agents):
            filename = './save_stat/' + name + '_stat.csv'
        
            bestY = []
            getgoal = []
            step = []
            step_per_record = []
            step_for_goal = []

            with open(filename, 'r') as f:
                read = csv.reader(f)
                for i, row in enumerate(read):
                    if i == 2400:
                        break
                    besty = float(row[3])
                    t = float(row[1])
                    bestY.append(besty)
                    step.append(t)
                    is_goal = 1 if besty >= 57 else 0
                    if i == 0:
                        sfg = 0
                    else:
                        sfg = t if is_goal else step_for_goal[i-1]
                    getgoal.append(is_goal)
                    
                    step_per_record.append(t / besty if besty > 1 else t)
                    step_for_goal.append(sfg)
                episodes = [i for i in range(len(bestY))]
                metrics = [bestY, getgoal, step, step_per_record, step_for_goal]
            
                drawall(name, episodes, metrics[idx], markers[ix], args.n, args.begin)
            
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=6, fontsize='x-large')
        plt.savefig(dir + '/result_' + ylabel + '.png')