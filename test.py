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

a = [0.5, 0.3, 0.8]
b = [0.4, 0,2, 0.6]

if a < b:
    print("a < b")
