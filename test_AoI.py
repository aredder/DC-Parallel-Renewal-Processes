import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pareto
from sklearn.datasets import make_spd_matrix
import random
import pandas as pd
import seaborn as sb

sb.set_theme()
sb.set_style("darkgrid")


class Worker:
    def __init__(self, coordinate, N_coordinates, pareto_exp):
        self.coordinate = coordinate
        self.pareto_exp = pareto_exp
        self.inv_mean_proc = 1/pareto.stats(self.pareto_exp, moments='m')
        self.proc_time = pareto.rvs(self.pareto_exp)  # initial processing time
        self.indices = np.zeros(N_coordinates)

    def get_job(self, new_indices):
        self.proc_time += pareto.rvs(self.pareto_exp)
        self.indices[:] = new_indices[:]


def pred_mean_aoi(workers, hp):
    n_cords = len(hp['K'])
    sum_of_invs = np.zeros(n_cords)
    for worker in workers:
        sum_of_invs[worker.coordinate] += worker.inv_mean_proc
    return np.einsum('i,j->ij', sum_of_invs, np.reciprocal(sum_of_invs))


def experiment(hp):
    n_cords = len(hp['K'])
    coordinate_index = np.zeros(n_cords, dtype=int)
    aoi_matrix = np.zeros((hp['time_steps'], n_cords, n_cords))
    workers = []
    for c, K in enumerate(hp['K']):
        workers.extend([Worker(c, n_cords, hp['q'][c]) for _ in range(K)])

    n = 0  # main iteration index
    while n < hp['time_steps']:

        k = np.argmin(np.array([worker.proc_time for worker in workers]))
        c = workers[k].coordinate

        aoi_matrix[coordinate_index[c], :, c] = coordinate_index - workers[k].indices
        coordinate_index[c] += 1
        workers[k].get_job(coordinate_index)

        n += 1
    return aoi_matrix, pred_mean_aoi(workers, hp)


def main():

    hp = {'K': [50, 35, 20],  # Number of workers per coordinate
          'q': [2.1, 2.8, 3.5],  # pareto exponent
          'time_steps': 100000,
          }
    runs = 1

    np.random.seed(30082023)
    random.seed(30082023)
    aoi_traj, pred_aoi_matrix = experiment(hp)

    n_cords = len(hp['K'])

    for c2 in range(n_cords):
        plt.figure()
        ax = plt.gca()
        for c1 in range(n_cords):
            df_aoi = pd.DataFrame(np.trim_zeros(aoi_traj[:, c1, c2]))
            cum_avg = df_aoi.expanding().mean()

            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(cum_avg, label=r'$\overline{\tau}_{' + str(c1+1) + str(c2+1) + '}(n)$', color=color)

            if c1 != c2:
                plt.hlines(pred_aoi_matrix[c1, c2]*(hp['K'][c2]), -1500, df_aoi.size + 1500, color=color,
                           label=r'$ \lim_{n\to \infty}\mathbb{E} [ \tau_{' + str(c1+1) + str(c2+1) + '}(n) ]$',
                           linestyles='--')
            else:
                plt.hlines(pred_aoi_matrix[c1, c2] * (hp['K'][c2]-1), -1500, df_aoi.size + 1500, color=color,
                           label=r'$ \lim_{n\to \infty}\mathbb{E} [ \tau_{' + str(c1+1) + str(c2+1) + '}(n) ]$',
                           linestyles='--')
        plt.xlabel(r'$n$')
        plt.legend(ncol=2, loc='lower right')
        plt.savefig('aoi_c' + str(c2+1) + '.pdf')
    plt.show()


if __name__ == "__main__":
    main()
