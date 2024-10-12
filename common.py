import math
import numpy as np

# 网络参数
from matplotlib import pyplot as plt

from deploy.grad.algorithm.util import point, IsIntersec

length = 67.1  # 区域边长
width = 22.6
g_size = 2  # 离散粒度
min_rssi = -60
nil_rssi = -100.0
power = 14
loss = 5
z_ap = 3

num = 6  # 节点个数
maxgen = 10  # 迭代次数
sizepop = 30  # 种群规模

obs = [
    {'id': 'ob02', 'loss': loss, 'x_start': 15, 'y_start': 0, 'x_end': 15, 'y_end': 9.5}
    , {'id': 'ob03', 'loss': loss, 'x_start': 30, 'y_start': 0, 'x_end': 30, 'y_end': 9.5}
    , {'id': 'ob04', 'loss': loss, 'x_start': 45, 'y_start': 0, 'x_end': 45, 'y_end': 9.5}
    , {'id': 'ob05', 'loss': loss, 'x_start': 60, 'y_start': 0, 'x_end': 60, 'y_end': 9.5}

    , {'id': 'ob08', 'loss': loss, 'x_start': 0, 'y_start': 9.5, 'x_end': 67.1, 'y_end': 9.5}
    , {'id': 'ob08', 'loss': loss, 'x_start': 15, 'y_start': 13.1, 'x_end': 60, 'y_end': 13.1}

    , {'id': 'ob10', 'loss': loss, 'x_start': 15, 'y_start': 13.1, 'x_end': 15, 'y_end': 22.6}
    , {'id': 'ob11', 'loss': loss, 'x_start': 37.5, 'y_start': 13.1, 'x_end': 37.5, 'y_end': 22.6}
    , {'id': 'ob13', 'loss': loss, 'x_start': 60, 'y_start': 13.1, 'x_end': 60, 'y_end': 22.6}
]


def list_to_string(list):
    if not list:
        return "null null"
    else:
        return str(round(np.min(list), 2)) + " " + str(round(np.average(list), 2))


def get_average_value(list):
    if not list:
        return nil_rssi
    else:
        return round(np.average(list), 2)

def get_min_value(list):
    if not list:
        return nil_rssi
    else:
        return round(np.min(list), 2)

def get_value(value, index):
    return str(round(np.average(np.array(value)[:, index]), 2))

def get_std_value(value, index):
    return str(round(np.std(np.array(value)[:, index]), 2))

def compute_signal(X_AP, Y_AP):
    rows = math.floor(width / g_size) + 1
    cols = math.floor(length / g_size) + 1

    grids = []
    overlap = 0
    l_r1 = []
    l_r2 = []
    l_r3 = []
    l_r4 = []
    l_all = []
    c_r1_s = 0
    c_r1_a = 0
    c_r2_s = 0
    c_r2_a = 0
    c_r3_s = 0
    c_r3_a = 0

    for row in range(rows):
        row_grid_rssi = []
        for col in range(cols):
            row_grid = g_size * row
            col_grid = g_size * col
            cur_rssi = nil_rssi
            for n in range(num):
                x_ap = X_AP[n]
                y_ap = Y_AP[n]
                distance = ((col_grid - x_ap) ** 2 + (row_grid - y_ap) ** 2 + (0 - z_ap) ** 2) ** 0.5
                # 计算频段损耗和路径损耗
                path_loss = 40 + math.log10(5.2 / 2.4) + 20 * math.log10(distance)
                wall_loss = 0
                # 障碍物损耗
                for l in range(len(obs)):
                    loss = obs[l]['loss']
                    x_start = obs[l]['x_start']
                    y_start = obs[l]['y_start']
                    x_end = obs[l]['x_end']
                    y_end = obs[l]['y_end']
                    # 计算墙体损耗
                    l1_start = point(col_grid, row_grid)
                    l1_end = point(x_ap, y_ap)
                    l2_start = point(x_start, y_start)
                    l2_end = point(x_end, y_end)
                    isIntersec = IsIntersec(l1_start, l1_end, l2_start, l2_end)
                    if (isIntersec):
                        wall_loss = wall_loss + loss
                # 该点的RSSI信号
                path_loss = path_loss + wall_loss
                rssi = power - path_loss
                if rssi > cur_rssi:
                    cur_rssi = rssi
                if rssi < min_rssi:
                    continue

            row_grid_rssi.append(cur_rssi)
            # 设置奖励
            if col_grid > 60 and col_grid <= length and row_grid >= 0 and row_grid <= 9.5:
                l_r1.append(cur_rssi)
                l_all.append(cur_rssi)
                c_r1_a = c_r1_a + 1
                if cur_rssi >= min_rssi:
                    c_r1_s = c_r1_s + 1
            elif col_grid >= 0 and col_grid <= 60 and row_grid >= 0 and row_grid <= 9.5:
                l_r2.append(cur_rssi)
                l_all.append(cur_rssi)
                c_r2_a = c_r2_a + 1
                if cur_rssi >= min_rssi:
                    c_r2_s = c_r2_s + 1
            elif col_grid > 15 and col_grid <= 60 and row_grid > 13.1 and row_grid <= width:
                l_r3.append(cur_rssi)
                l_all.append(cur_rssi)
                c_r3_a = c_r3_a + 1
                if cur_rssi >= min_rssi:
                    c_r3_s = c_r3_s + 1
            # 设置惩罚
            elif col_grid >= 0 and col_grid <= 15 and row_grid > 13.1 and row_grid <= width:
                l_r4.append(cur_rssi)
            elif col_grid >= 0 and col_grid <= length and row_grid > 9.5 and row_grid <= 13.1:
                l_r4.append(cur_rssi)
            elif col_grid > 60 and col_grid <= length and row_grid > 13.1 and row_grid <= width:
                l_r4.append(cur_rssi)
            overlap = overlap + 1 / cur_rssi
        grids.append(row_grid_rssi)

    record_str = list_to_string(l_r1) + " " + list_to_string(l_r2) + " " + list_to_string(
        l_r3) + " " + list_to_string(l_r4) + " " + list_to_string(l_all) + "  coverage: " + str(
        round(c_r1_s / c_r1_a, 2)) + " " + str(round(c_r2_s / c_r2_a, 2)) + " " + str(
        round(c_r3_s / c_r3_a, 2)) + " " + str(round((c_r1_s + c_r2_s + c_r3_s) / (c_r1_a + c_r2_a + c_r3_a), 2))
    rss = []
    rss.append(get_average_value(l_r1))
    rss.append(get_average_value(l_r2))
    rss.append(get_average_value(l_r3))
    rss.append(get_average_value(l_r4))
    rss.append(get_average_value(l_all))
    rss.append(get_min_value(l_r1))
    rss.append(get_min_value(l_r2))
    rss.append(get_min_value(l_r3))
    rss.append(get_min_value(l_r4))
    rss.append(get_min_value(l_all))
    cov = []
    cov.append(round(c_r1_s / c_r1_a, 2))
    cov.append(round(c_r2_s / c_r2_a, 2))
    cov.append(round(c_r3_s / c_r3_a, 2))
    cov.append(round((c_r1_s + c_r2_s + c_r3_s) / (c_r1_a + c_r2_a + c_r3_a), 2))

    # print(record_str)
    return grids, record_str, overlap, rss, cov


def print_map(grid, X_AP, Y_AP):
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.xlim(0, length)
    plt.ylim(0, width)
    plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, hspace=0.23)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] > min_rssi:
                rect = plt.Rectangle((j * g_size, i * g_size), g_size, g_size, lw=0.5, color='y', ec='black', fill=True)
                plt.gcf().gca().add_patch(rect)
                # plt.gcf().gca().text(j * G_size + G_size * 0.5, i * G_size + G_size * 0.5, bestGrids[i][j], fontsize=4, ha='center',
                #              va='center')
        for n in range(num):
            circle = plt.Circle((X_AP[n], Y_AP[n]), 0.5, color='b')
            plt.gcf().gca().add_artist(circle)
    plt.show()
