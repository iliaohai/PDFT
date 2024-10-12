import numpy as np
import matplotlib.pyplot as plt
import time

from deploy.grad.algorithm.common import sizepop, num, compute_signal, maxgen, length, width, g_size, print_map, \
    get_value, get_std_value
from deploy.grad.algorithm.proposed_common import init_circles, optimize_circles


def get_zoa_location():
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


def random_zoa_position(x1, xb, y1, yb, r1, r2):
    count = 0
    while (count < 50):
        count = count + 1
        x = x1 + r1 * (xb - x1) - r2 * (x1 - xb)
        y = y1 + r1 * (yb - y1) - r2 * (y1 - yb)
        if x >= 0 and x <= length and y >= 0 and y <= 6.7:
            return x, y
        elif x > 15 and x <= 60 and y > 13.1 and y <= width:
            return x, y

    return x, y

ts = []
rss_iter = []
cov_iter = []
for i in range(5):
    X_ALL = []
    Y_ALL = []
    for s in range(sizepop):
        X_AP = []
        Y_AP = []
        origin_circles, bounds, radii = init_circles(num)
        uniform_circles = optimize_circles(origin_circles, display=False)
        for n in range(len(uniform_circles)):
            x = uniform_circles[n][0]
            y = uniform_circles[n][1]
            X_AP.append(x)
            Y_AP.append(y)
        X_ALL.append(X_AP)
        Y_ALL.append(Y_AP)

    x_best_position = X_ALL[0]
    y_best_position = Y_ALL[0]
    best_grids, best_record, best_overlap, best_rss, best_cov = compute_signal(x_best_position, y_best_position)

    start = time.time()
    for gen in range(maxgen):
        for s in range(sizepop):
            x_position = X_ALL[s]
            y_position = Y_ALL[s]
            grids, record, overlap, rss, cov = compute_signal(x_position, y_position)
            if overlap < best_overlap:
                best_grids = grids
                best_overlap = overlap
                best_record = record
                best_rss = rss
                best_cov = cov
                x_best_position = x_position
                y_best_position = y_position
                # print('round: ', gen, 'coverage: ', bestCoverage, 'total', total)
                # print(record)

        r1, r2 = np.random.rand(), np.random.rand()
        X_Tmp = []
        Y_Tmp = []
        for s in range(sizepop):
            X_AP = []
            Y_AP = []
            x_position = X_ALL[s]
            y_position = Y_ALL[s]
            for n in range(num):
                x, y = random_zoa_position(x_position[n], x_best_position[n], y_position[n], y_best_position[n], r1, r2)
                X_AP.append(x)
                Y_AP.append(y)
            X_Tmp.append(X_AP)
            Y_Tmp.append(Y_AP)
        X_ALL = X_Tmp
        Y_ALL = Y_Tmp

        # print(gen, ":", best_overlap)

    end = time.time()
    run_time = end - start
    #print("run time:", round(run_time, 2))
    ts.append(round(run_time, 2))
    rss_iter.append(best_rss)
    cov_iter.append(best_cov)
    print('best signal', best_record)
print(round(np.average(ts), 2))
print(get_value(cov_iter, 0) + " " + get_std_value(cov_iter, 0) + " " + get_value(rss_iter, 0) + " " + get_std_value(
    rss_iter, 0) + " " + get_value(cov_iter, 3) + " " + get_std_value(cov_iter, 3) + " " + get_value(rss_iter, 4)
      + " " + get_std_value(rss_iter, 4) + " " + str(round(np.average(ts), 2)))
# print(get_value(rss_iter, 0)+" "+get_value(rss_iter, 1)+" "+get_value(rss_iter, 2)+" "+get_value(rss_iter, 3)
#       +" "+get_value(cov_iter, 0)+" "+get_value(cov_iter, 1)+" "+get_value(cov_iter, 2)+" "+get_value(cov_iter, 3))
# print(get_value(rss_iter, 5)+" "+get_value(rss_iter, 0)+" "+get_value(rss_iter, 6)+" "+get_value(rss_iter, 1)+" "+get_value(rss_iter, 7)
#       +" "+get_value(rss_iter, 2)+" "+get_value(rss_iter, 8)+" "+get_value(rss_iter, 3)+" "+get_value(rss_iter, 9)+" "+get_value(rss_iter, 4))

#print_map(best_grids, x_best_position, y_best_position)