import os
import pandas as pd
import numpy as np
import sys
import warnings
from collections import OrderedDict
import torch.nn.functional as F
from rank_cylib.rank_cy import evaluate_cy
import torch
import time

#IS_CYTHON_AVAI = True
'''
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython rank evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )
'''
def evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank=50,
        use_metric_cuhk03=False,
        use_cython=True,
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """

    return evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)

@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()

read_time = time.time()
filename_q = sys.argv[1]      #"./output_query_feature.txt"
filename_g = sys.argv[2]      #"./output_gallery_feature.txt"
f_q = open(filename_q,"r")
f_g = open(filename_g,"r")

listlines_q = f_q .readlines()
listlines_g = f_g .readlines()
m = len(listlines_q)
n = len(listlines_g)
qf,gf,q_pids,g_pids,q_camids,g_camids= list(),list(),list(),list(),list(),list()
for i in range(m):
    a = listlines_q[i].split()[3:]
    a=list(map(float,a))
    qf.append(a)
    b = listlines_q[i].split()[1:2]
    q_pids.append(b[0])
    c = listlines_q[i].split()[2:3]
    
    q_camids.append(int(c[0][1:]))


for i in range(n):
    a = listlines_g[i].split()[3:]
    a=list(map(float,a))
    gf.append(a)
    b = listlines_g[i].split()[1:2]
    g_pids.append(b[0])
    c = listlines_g[i].split()[2:3]
    g_camids.append(int(c[0][1:]))

qf = torch.Tensor(qf).cpu()
gf = torch.Tensor(gf).cpu()

q_pids = np.asarray(q_pids)
g_pids = np.asarray(g_pids)
q_camids = np.asarray(q_camids)
g_camids = np.asarray(g_camids)
print(f"load data using {time.time()- read_time} seconds!")

'''
distmat = np.repeat(np.sum(np.power(qf,2), axis=1, keepdims=True), n, axis=1) + \
                   np.repeat(np.sum(np.power(gf,2), axis=1, keepdims=True), m, axis=1) .T
distmat = distmat - 2*(qf.dot(gf.T))

cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
print("Results ----------")
print("mAP: {:.1%}".format(mAP))
print("CMC curve")
for r in [1, 5, 10]:
	print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
print("------------------")
'''

dist_time = time.time()
distmat = compute_cosine_distance(qf, gf)
print(f"calculate distmat using {time.time()- dist_time} seconds!")

# print(distmat.shape, type(distmat))
# print(q_pids.shape, type(q_pids))
# print(g_pids.shape, type(g_pids))
# print(q_camids.shape, type(q_camids))
# print(g_camids.shape, type(g_camids))
# print(q_pids)
cmc, all_AP, all_INP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, 10, False, True)
mAP = np.mean(all_AP)
mINP = np.mean(all_INP)
for r in [1, 5, 10]:
    print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))
print('mAP', mAP * 100)
print('mINP', mINP * 100)
print("metric" , (mAP + cmc[0]) / 2 * 100)

f_g.close()
f_q.close()
