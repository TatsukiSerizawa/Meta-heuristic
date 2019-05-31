import random
import numpy as np

'''
N = 10 # 粒子数

# 最小値，最大値設定
x_min, x_max = -5, 5
y_min, y_max = -5, 5

test = []

ps = [{"x": random.uniform(x_min, x_max), 
    "y": random.uniform(y_min, y_max)} for i in range(N)]

for i in range(N):
    test.append({"x": random.uniform(x_min, x_max),
    "y": random.uniform(y_min, y_max)})

print(ps)
print(test)
'''


search_space_x, search_space_y = {'min': 2.0, "max": 6.0}, {"min": 1.0, "max": 5.0}

print(search_space_x["min"])

a = b = 1

print(a)
print(b)