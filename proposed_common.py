import math
import random
import time

import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from scipy.optimize import minimize
from PIL import Image

from deploy.grad.algorithm.common import length, width, g_size, nil_rssi, z_ap, obs, power, min_rssi, list_to_string, \
    get_average_value, get_min_value
from deploy.grad.common_utils import point, IsIntersec

r_size = 0.1  # 热图显示
vmin = -60
vmax = -30
is_loss = 1
epochs = 10
dpi = 600
new_size = (671, 226)
record_str = ""


def init_circles(num):
    # 初始化圆的位置
    r = math.ceil(math.pow((length * width / 3.14 / num), 0.5))
    radii = [r for _ in range(num)]
    origin_circles = [(random.uniform(0, length), random.uniform(0, width), r) for r in radii]
    bounds = []
    for r in radii:
        bounds.extend([(0, length), (0, width)])
    return origin_circles, bounds, radii


def optimize_circles_bfgs(circles, bounds, radii, display=True):
    """使用BFGS梯度优化方法来优化圆的位置。"""
    # 将圆心坐标转换为一维数组
    initial_coords = np.array([c for circle in circles for c in circle[:2]])

    # 定义一个闭包来计算总重叠面积
    def loss(coords):
        return total_overlap_vectorized(coords, display, False)

    # 使用BFGS算法优化圆的位置
    optimized_result = minimize(loss, initial_coords, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 10, 'disp': display})
    optimized_coords = optimized_result.x

    # 将优化后的坐标转换回 (x, y, r) 格式
    optimized_circles = [(optimized_coords[2 * i], optimized_coords[2 * i + 1], radii[i]) for i in range(len(radii))]
    return optimized_circles


def calculate_overlap(c1, c2):
    """计算两个圆的重叠面积。"""
    d = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
    r1, r2 = c1[2], c2[2]

    # 如果圆心距离大于半径之和，则不重叠
    if d >= r1 + r2:
        return 0.0

    # 如果其中一个圆完全包含另一个圆，则重叠面积为较小圆的面积
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2

    # 使用圆的重叠面积公式计算
    a = r1 ** 2 * math.acos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    b = r2 ** 2 * math.acos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    x = (d ** 2 - r2 ** 2 + r1 ** 2) / (2 * d)
    z = x ** 2
    y = math.sqrt(r1 ** 2 - z)

    return a + b - d * y


def total_overlap(circles):
    """
    计算所有圆的总重叠面积，包含与矩形边界的重叠。
    """
    overlap = 0.0

    # 计算圆之间的重叠
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            overlap += calculate_overlap(circles[i], circles[j])

    return overlap


def optimize_circles(circles, iterations=10, display=True):
    """优化圆的位置并计算每次迭代的重叠面积。"""
    if display == True:
        initial_coords = np.array([c for circle in circles for c in circle[:2]])
        total_overlap_vectorized(initial_coords)

    for iteration in range(iterations):
        for i, (x, y, r) in enumerate(circles):
            min_overlap = float('inf')
            best_position = (x, y)

            # 对每一个圆，遍历所有坐标查找最优位置
            for new_x in np.linspace(1 / math.sqrt(2) * r, length - 1 / math.sqrt(2) * r, 20):  # 20是步长，可以根据需要调整
                for new_y in np.linspace(1 / math.sqrt(2) * r, width - 1 / math.sqrt(2) * r, 20):
                    if new_y >= 9.5 and new_y < 13.1:
                        continue
                    if new_x >= 0 and new_x < 15 and new_y >= 13.1 and new_y < width:
                        continue
                    if new_x >= 60 and new_x < length and new_y >= 13.1 and new_y < width:
                        continue
                    circles[i] = (new_x, new_y, r)
                    current_overlap = total_overlap(circles)
                    if current_overlap < min_overlap:
                        min_overlap = current_overlap
                        best_position = (new_x, new_y)

            circles[i] = best_position + (r,)

        # 计算每次迭代后的总重叠面积
        # print(f"Iteration {iteration + 1}, Total Overlap: {total_overlap(circles)}")
        if display == True:
            initial_coords = np.array([c for circle in circles for c in circle[:2]])
            total_overlap_vectorized(initial_coords)

    return circles


# 可视化最终的圆
def print_circles(circles):
    fig, ax = plt.subplots(figsize=(3.5, 1.22))
    plt.subplots_adjust(left=0.01, bottom=0.02, right=0.99, top=0.98, hspace=0.23)
    bk_img = Image.open('./fig/room2-2.jpg')
    bk_img = bk_img.resize(new_size)
    ax.imshow(bk_img)
    plt.axis('off')

    ax.set_xlim(0, math.floor(length / r_size))
    ax.set_ylim(0, math.floor(width / r_size))

    for x, y, r in circles:
        circle = plt.Circle((math.floor(x / r_size), math.floor(y / r_size)), math.floor(r / r_size), linewidth=0.8,
                            edgecolor='b', alpha=0.2)
        ax.add_patch(circle)

    plt.savefig("../out/overlap.jpg", dpi=dpi)
    # plt.show()


# 可视化最终的圆
def print_circles1(circles):
    fig, ax = plt.subplots()
    rectangle = plt.Rectangle((0, 0), length, width, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)

    for x, y, r in circles:
        circle = plt.Circle((x, y), r, linewidth=1, edgecolor='b', facecolor='blue', alpha=0.2)
        ax.add_patch(circle)

    plt.xlim(0, length)
    plt.ylim(0, width)
    ax.set_aspect(1)
    plt.show()


rows = math.floor(width / g_size) + 1
cols = math.floor(length / g_size) + 1


def total_overlap_vectorized(circles, display=True, data=False):
    overlap = 0.0
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
        row_grid = g_size * row
        for col in range(cols):
            col_grid = g_size * col
            cur_rssi = nil_rssi
            # 计算所有AP在该点的信号值
            num_circles = len(circles) // 2
            c = 0
            for i in range(num_circles):
                x_ap = circles[2 * i]
                y_ap = circles[2 * i + 1]
                distance = ((col_grid - x_ap) ** 2 + (row_grid - y_ap) ** 2 + (0 - z_ap) ** 2) ** 0.5
                # 计算频段损耗和路径损耗
                path_loss = 40 + math.log10(5.2 / 2.4) + 20 * math.log10(distance)
                # 计算墙体损耗
                wall_loss = 0
                # 障碍物损耗
                if is_loss == 1:
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

            reward = 1
            # 设置奖励
            if col_grid > 60 and col_grid <= length and row_grid >= 0 and row_grid <= 9.5:
                c_r1_a = c_r1_a + 1
                l_r1.append(cur_rssi)
                l_all.append(cur_rssi)
                if cur_rssi >= min_rssi:
                    c_r1_s = c_r1_s + 1
                    reward = 300
            elif col_grid >= 0 and col_grid <= 60 and row_grid >= 0 and row_grid <= 9.5:
                c_r2_a = c_r2_a + 1
                l_r2.append(cur_rssi)
                l_all.append(cur_rssi)
                if cur_rssi >= min_rssi:
                    c_r2_s = c_r2_s + 1
                    reward = 2
            elif col_grid > 15 and col_grid <= 60 and row_grid > 13.1 and row_grid <= width:
                c_r3_a = c_r3_a + 1
                l_r3.append(cur_rssi)
                l_all.append(cur_rssi)
                if cur_rssi >= min_rssi:
                    c_r3_s = c_r3_s + 1
                    reward = 10
            # 设置惩罚
            elif col_grid >= 0 and col_grid <= 15 and row_grid > 13.1 and row_grid <= width:
                l_r4.append(cur_rssi)
                if cur_rssi >= min_rssi:
                    reward = 0.1
            elif col_grid >= 0 and col_grid <= length and row_grid > 9.5 and row_grid <= 13.1:
                l_r4.append(cur_rssi)
                if cur_rssi >= min_rssi:
                    reward = 0.1
            elif col_grid > 60 and col_grid <= length and row_grid > 13.1 and row_grid <= width:
                l_r4.append(cur_rssi)
                if cur_rssi >= min_rssi:
                    reward = 0.1
            if cur_rssi >= min_rssi:
                overlap = overlap + reward / cur_rssi

    # print(temp)
    # coverage = count / (rows * cols)
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

    if display == True:
        print(record_str)

    if data == True:
        return rss, cov

    return overlap

def caculate_final_signal(coords):
    rows = math.floor(width / r_size) + 1
    cols = math.floor(length / r_size) + 1
    grids = []
    count = 0
    for row in range(rows):
        row_grid_rssi = []
        row_grid = r_size * row
        for col in range(cols):
            col_grid = r_size * col
            cur_rssi = nil_rssi
            # 计算所有AP在该点的信号值
            for x, y, r in coords:
                distance = ((col_grid - x) ** 2 + (row_grid - y) ** 2 + (0 - z_ap) ** 2) ** 0.5
                # 计算频段损耗和路径损耗
                path_loss = 40 + math.log10(5.2 / 2.4) + 20 * math.log10(distance)
                # 计算墙体损耗
                wall_loss = 0
                # 障碍物损耗
                if is_loss == 1:
                    for l in range(len(obs)):
                        loss = obs[l]['loss']
                        x_start = obs[l]['x_start']
                        y_start = obs[l]['y_start']
                        x_end = obs[l]['x_end']
                        y_end = obs[l]['y_end']
                        # 计算墙体损耗
                        l1_start = point(col_grid, row_grid)
                        l1_end = point(x, y)
                        l2_start = point(x_start, y_start)
                        l2_end = point(x_end, y_end)
                        isIntersec = IsIntersec(l1_start, l1_end, l2_start, l2_end)
                        if (isIntersec):
                            wall_loss = wall_loss + loss
                # 该点的RSSI信号
                path_loss = path_loss + wall_loss
                rssi = power - path_loss
                if rssi > cur_rssi and rssi >= min_rssi:
                    cur_rssi = round(rssi, 3)
            if cur_rssi >= min_rssi:
                count = count + 1
            row_grid_rssi.append(cur_rssi)
        grids.append(row_grid_rssi)
    coverage = round(count / (rows * cols), 2)
    return grids, coverage


def print_many_aps(loc9, filename, display=True):
    for i in range(9):
        if i != 8:
            origin_circles, bounds, radii = init_circles(i + 1)
            uniform_circles = optimize_circles(origin_circles, display=False)
            optimized_circles = optimize_circles_bfgs(uniform_circles, bounds, radii, display)
        else:
            optimized_circles = loc9
        bk_img = Image.open('fig/room2-1.jpg')
        bk_img = bk_img.resize(new_size)
        fr_img = Image.open('fig/ap.png')
        fr_img = fr_img.resize((32, 32))
        # 显示AP的位置
        for x, y, r in optimized_circles:
            px = math.floor(x / r_size) - 16
            py = 226 - math.floor(y / r_size) - 16
            bk_img.paste(fr_img, (px, py), mask=fr_img)
        fn = "../out/"+filename + str(i+1) + ".jpg"
        bk_img.save(fn)


def print_comparsion(loc9, filename, display=True):
    origin_circles, bounds, radii = init_circles(9)
    optimized_circles = optimize_circles_bfgs(origin_circles, bounds, radii, display=display)
    bk_img = Image.open('fig/room2-1.jpg')
    bk_img = bk_img.resize(new_size)
    fr_img = Image.open('fig/ap.png')
    fr_img = fr_img.resize((24, 24))
    # 显示AP的位置
    for x, y, r in optimized_circles:
        px = math.floor(x / r_size) - 12
        py = 226 - math.floor(y / r_size) - 12
        bk_img.paste(fr_img, (px, py), mask=fr_img)
    fn = "../out/" + filename + "-a.jpg"
    bk_img.save(fn)

    optimized_circles = loc9
    bk_img = Image.open('fig/room2-1.jpg')
    bk_img = bk_img.resize(new_size)
    fr_img = Image.open('fig/ap.png')
    fr_img = fr_img.resize((24, 24))
    # 显示AP的位置
    for x, y, r in optimized_circles:
        px = math.floor(x / r_size) - 12
        py = 226 - math.floor(y / r_size) - 12
        bk_img.paste(fr_img, (px, py), mask=fr_img)
    fn = "../out/" + filename + "-b.jpg"
    bk_img.save(fn)


def print_heatmap(grids, circles, filename):
    fig, ax1 = plt.subplots(figsize=(3.5, 1.18))
    plt.subplots_adjust(left=0.04, bottom=0.168, right=1.02, top=0.94, hspace=0.23)
    bk_img = Image.open('./fig/room2-2.jpg')
    bk_img = bk_img.resize(new_size)
    ax1.imshow(bk_img)

    ax1.tick_params(width=0.5, length=2)
    ax1.spines['right'].set_linewidth(0.5)
    ax1.spines['top'].set_linewidth(0.5)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['bottom'].set_linewidth(0.5)
    ax1.set_xlim(0, math.floor(length / r_size))
    ax1.set_ylim(0, math.floor(width / r_size))
    plt.xticks([0, 100, 200, 300, 400, 500, 600], ('0m', '10m', '20m', '30m', '40m', '50m', '60m'), fontsize=5)
    plt.yticks([0, 50, 100, 150, 200], ('0m', '5m', '10m', '15m', '20m'), fontsize=5)

    levels = np.linspace(vmin, vmax, 400 + 1)
    contourf_ = ax1.contourf(grids, levels=levels, cmap='viridis', alpha=0.5)  # 8表示要分几部分等高线
    cbar = fig.colorbar(contourf_, pad=0.02)
    # cbar.set_ticks([-25, -30, -35, -40, -45, -50, -55, -60])
    # cbar.set_ticklabels(['-25dBm', '-30dBm', '-35dBm', '-40dBm', '-45dBm', '-50dBm', '-55dBm', '-60dBm'], fontsize=5)
    cbar.set_ticks([-30, -40, -50, -60])
    cbar.set_ticklabels(['-30dBm', '-40dBm', '-50dBm', '-60dBm'], fontsize=5)
    cbar.outline.set_linewidth(0.8)
    ax2 = cbar.ax
    ax2.tick_params(width=0.5, length=2)

    plt.savefig(filename, dpi=dpi)
    # plt.show()


def print_num_aps(num, filename, display=True, data=False):
    # 随机产生AP
    start = time.time()
    origin_circles, bounds, radii = init_circles(num)
    # 下一行为不需要PD，直接计算FT
    # initial_coords = np.array([c for circle in uniform_circles for c in circle[:2]])
    # optimized_circles = optimize_circles_bfgs(origin_circles, bounds, radii, display=display)
    # 初始化AP位置
    uniform_circles = optimize_circles(origin_circles, display=display)
    if(display == True):
        print("end of overlap calculate")
        print_circles(uniform_circles)
    # 优化AP位置
    optimized_circles = optimize_circles_bfgs(uniform_circles, bounds, radii, display=display)
    end = time.time()
    run_time = end - start
    # print("run time:", round(run_time, 2))
    rss = []
    cov = []
    if display == False:
        initial_coords = np.array([c for circle in optimized_circles for c in circle[:2]])
        rss, cov = total_overlap_vectorized(initial_coords, display=False, data=True)
    # print_circles(optimized_circles)
    # 显示热图
    # 绘制热图
    if display == True:
        grids, coverage = caculate_final_signal(optimized_circles)
        print_heatmap(grids, optimized_circles, filename)

    if data == True:
        return rss, cov, run_time
    return optimized_circles
