# 1. 在指定背景图部署9个AP
import numpy as np

from deploy.grad.algorithm.common import get_value, get_std_value
from deploy.grad.algorithm.proposed_common import print_num_aps, print_comparsion, print_many_aps
# for i in range(10):
#     loc3 = print_num_aps(7, "./fig/out-heatmap.jpg", display=False)
# loc3 = print_num_aps(3, "../out/heatmap.jpg", display=False)
# print("3 aps end ====================")
# loc6 = print_num_aps(6, "../out/heatmap.jpg", display=False)
# print("6 aps end ====================")
# ts = []
# rss_iter = []
# cov_iter = []
# for i in range(5):
#     rss, cov, run_time = print_num_aps(6, "../out/heatmap.jpg", display=False, data=True)
#     rss_iter.append(rss)
#     cov_iter.append(cov)
#     ts.append(run_time)
#
# # print(round(np.average(ts),2))
# # print(get_value(cov_iter, 0) + " " + get_std_value(cov_iter, 0) + " " + get_value(rss_iter, 0) + " " + get_std_value(
# #     rss_iter, 0) + " " + get_value(cov_iter, 3) + " " + get_std_value(cov_iter, 3) + " " + get_value(rss_iter, 4)
# #       + " " + get_std_value(rss_iter, 4) + " " + str(round(np.average(ts), 2)))
#
# # print(get_value(rss_iter, 0)+" "+get_value(rss_iter, 1)+" "+get_value(rss_iter, 2)+" "+get_value(rss_iter, 3)
# #       +" "+get_value(cov_iter, 0)+" "+get_value(cov_iter, 1)+" "+get_value(cov_iter, 2)+" "+get_value(cov_iter, 3)+ " " + str(round(np.average(ts),2)))
# print(get_value(rss_iter, 5)+" "+get_value(rss_iter, 0)+" "+get_value(rss_iter, 6)+" "+get_value(rss_iter, 1)+" "+get_value(rss_iter, 7)
#       +" "+get_value(rss_iter, 2)+" "+get_value(rss_iter, 8)+" "+get_value(rss_iter, 3)+" "+get_value(rss_iter, 9)+" "+get_value(rss_iter, 4))

loc9 = print_num_aps(9, "../out/heatmap.jpg", display=True)
print("9 aps end ====================")

# 2. 比较在不需要预部署AP位置和需要预部署AP位置的差别
print_comparsion(loc9, "ablation", display=False)
print("ablation aps end ====================")

# 1-9个AP，计算不同AP数量，在同一张地图中的规划位置
print_many_aps(loc9, "manyaps", display=False)
print("many aps end ====================")