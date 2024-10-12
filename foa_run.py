import math
import numpy as np
import matplotlib.pyplot as plt
import time

from deploy.grad.FOA.util import point, IsIntersec
from deploy.grad.algorithm.common import length, width, maxgen, sizepop, num, compute_signal, g_size, print_map, \
    get_value, get_std_value

s = 1  # 步长

# 随机初始化果蝇群体位置

# 个体和速度最大和最小值

def get_foa_location():
    count = 0
    while (count < 50):
        count = count + 1
        x = length * np.random.rand()
        y = width * np.random.rand()
        if x >= 0 and x <= length and y >= 0 and y <= 6.7:
            return x, y
        elif x > 15 and x <= 60 and y > 13.1 and y <= width:
            return x, y

    return x, y


def random_foa_position(i1, i2):
    count = 0
    while (count < 50):
        count = count + 1
        x = i1 + 2 * s * np.random.rand() - s
        y = i2 + 2 * s * np.random.rand() - s
        if x >= 0 and x <= length and y >= 0 and y <= 6.7:
            return x, y
        elif x > 15 and x <= 60 and y > 13.1 and y <= width:
            return x, y

    return x, y

ts = []
rss_iter = []
cov_iter = []
for i in range(5):
    best_coverage = 0
    best_overlap = 0
    best_grids = []
    best_record = ""

    X_AP = []
    Y_AP = []

    for n in range(num):
        x, y = get_foa_location()
        X_AP.append(x)
        Y_AP.append(y)

    start = time.time()
    for gen in range(maxgen):
        for i in range(sizepop):
            # 随机位置
            X = []
            Y = []
            for n in range(num):
                x, y = random_foa_position(X_AP[n], Y_AP[n])
                X.append(x)
                Y.append(y)

            # 味道浓度函数(覆盖率)
            grids, record, overlap, rss, cov = compute_signal(X, Y)
            if overlap < best_overlap:
                best_grids = grids
                best_overlap = overlap
                best_record = record
                best_rss = rss
                best_cov = cov
                X_AP = X
                Y_AP = Y
                # print('round: ', gen, 'coverage: ', bestCoverage, 'total', total)
                # print(record)

        #print(gen, ":", best_overlap)

    end = time.time()
    run_time = end - start
    # print("run time:", round(run_time, 2))
    # print('best signal', best_record, best_overlap)
    ts.append(round(run_time, 2))
    rss_iter.append(best_rss)
    cov_iter.append(best_cov)

    #print('best signal', best_record, best_overlap)


# print(round(np.average(ts),2))
# print(get_value(cov_iter, 0) + " " + get_std_value(cov_iter, 0) + " " + get_value(rss_iter, 0) + " " + get_std_value(
#     rss_iter, 0) + " " + get_value(cov_iter, 3) + " " + get_std_value(cov_iter, 3) + " " + get_value(rss_iter, 4)
#       + " " + get_std_value(rss_iter, 4) + " " + str(round(np.average(ts))))
# print(get_value(rss_iter, 0)+" "+get_value(rss_iter, 1)+" "+get_value(rss_iter, 2)+" "+get_value(rss_iter, 3)
#       +" "+get_value(cov_iter, 0)+" "+get_value(cov_iter, 1)+" "+get_value(cov_iter, 2)+" "+get_value(cov_iter, 3) + " " + str(round(np.average(ts), 2)))
print(get_value(rss_iter, 5)+" "+get_value(rss_iter, 0)+" "+get_value(rss_iter, 6)+" "+get_value(rss_iter, 1)+" "+get_value(rss_iter, 7)
      +" "+get_value(rss_iter, 2)+" "+get_value(rss_iter, 8)+" "+get_value(rss_iter, 3)+" "+get_value(rss_iter, 9)+" "+get_value(rss_iter, 4))

# print_map(best_grids, X_AP, Y_AP)