import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from cycler import cycler

import numpy as np
import seaborn as sb
from scipy.stats import expon, poisson
from sklearn.datasets import make_spd_matrix
import random
import pandas as pd

sb.set_theme()
sb.set_style("darkgrid")

class Worker:
    def __init__(self, coordinate, N_coordinates, lamb):
        self.coordinate = coordinate
        self.lamb = lamb
        self.inv_mean_proc = 1/expon.stats(scale=1/self.lamb, moments='m')
        self.proc_time = expon.rvs(scale=1/self.lamb)  # initial processing time
        self.indices = np.zeros(N_coordinates)

    def get_job(self, new_indices):
        self.proc_time += expon.rvs(scale=1/self.lamb)
        self.indices = new_indices


def pred_aoi_secondm(K, q):
    lamb = q
    mean = 1/lamb
    var = 1/(lamb**2)
    scd_m = 2/(lamb**2)
    sol = (K-1)*((scd_m/(2*mean))**2 + var*mean/(mean**3)) + scd_m*((K-1)/mean)**2

    return sol

def secondary_pred_aoi_secondm(K, q):
    lamb = q
    norm_N = (lamb + lamb**2)
    sol = (2/lamb**2 + 2/lamb + 1)*(K-1)**2*norm_N

    return sol

def exact_aoi_secondm(K, q):
    lamb = q
    mean = 1/lamb
    var = 1/(lamb**2)
    scd_m = 2/(lamb**2)
    sol = (K-1) + scd_m*((K-1)/mean)**2
    sol = (K-1)*(1+2*(K-1))

    return sol


def experiment(K, T, q):

    aoi = np.zeros(T)
    workers = [Worker(0, 1, q) for _ in range(K)]

    n = 0  # main iteration index
    while n < T:

        k = np.argmin(np.array([worker.proc_time for worker in workers]))
        aoi[n] = n - workers[k].indices
        n += 1
        workers[k].get_job(n)

    cum_second_moment = np.mean(np.square(aoi))
    prediction = pred_aoi_secondm(K, q)

    def rel_error(true, pred):
        return pred - true

    return rel_error(cum_second_moment, prediction)


def direct(K, q, mode):

    if mode == 0:
        prediction = pred_aoi_secondm(K, q)
        true = exact_aoi_secondm(K, q)

        def rel_error(true, pred):
            return (pred - true) / true

        return rel_error(true, prediction)
    else:
        true = exact_aoi_secondm(K, q)
        prediction1 = pred_aoi_secondm(K, q)
        prediction2 = secondary_pred_aoi_secondm(K, q)
        return (prediction1 - prediction2)



def main():
    np.random.seed(30082023)
    random.seed(30082023)



    dim = 1000

    Ks = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    qs = np.linspace(0.001, 3, dim)

    count = 0

    plt.figure()
    #

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--'])


    for i, K in enumerate(Ks):
        Z = []
        for q in qs:
            Z.append(direct(K, q, 0))
        count += 1
        Z = np.array(Z)

        plt.rc('axes', prop_cycle=linestyle_cycler)
        plt.plot(qs, Z, label=r'$K = ' + str(K) + '$', color=color_cycle[i])

    plt.xlabel(r'$\lambda = 1/\mu$')
    plt.ylabel('Bound Corollary 3 - Bound Prop. 5 ')
    plt.yscale('log')
    ax = plt.gca()
    ax.set_yticks(ax.get_yticks()[::2])
    plt.legend(ncol=2)
    plt.savefig('aoi_moment_error_computed.pdf')

    plt.show()





if __name__ == "__main__":
    main()
