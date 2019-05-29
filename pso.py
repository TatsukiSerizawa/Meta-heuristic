# coding: utf-8

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 評価関数(z = x^2+y^2)
def evaluation(x, y):
    #return x**2 + y**2 # Sphere function
    return (1.0 - x)**2 + 100 * (y - x**2)**2

# 粒子の位置更新関数
def update_position(x, y, vx, vy):
    new_x = x + vx
    new_y = y + vy
    return new_x, new_y

# 粒子の速度更新関数
def update_velocity(x, y, vx, vy, p, g, W, C1, C2):
    #random parameter (0~1)
    r1 = random.random()
    r2 = random.random()
    # 速度更新
    new_vx = W * vx + C1 * (g["x"] - x) * r1 + C2 * (p["x"] - x)
    new_vy = W * vy + C1 * (g["y"] - y) * r1 + C2 * (p["y"] - y)
    return new_vx, new_vy

# 可視化
def visualization(positions, SWARM_SIZE):
    fig = plt.figure()
    ax = Axes3D(fig)
    # Mesh
    mesh_x = np.arange(-2.0, 2.0, 0.1)
    mesh_y = np.arange(-2.0, 2.0, 0.1)
    mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)
    mesh_Z = evaluation(mesh_X, mesh_Y)
    ax.plot_wireframe(mesh_X, mesh_Y, mesh_Z)
    # Particle
    for j in range(SWARM_SIZE):
        z = evaluation(positions[j]["x"], positions[j]["y"])
        ax.scatter(positions[j]["x"], positions[j]["y"], z)
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.0, 3.0])
    ax.set_zlim([-10.0, 3000.0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.show()


def main():
    SWARM_SIZE = 100 # 粒子数
    ITERATION = 30 # ループ回数
    W = 0.5 # 慣性係数パラメータ
    C1 = 0.84 # 加速係数
    C2 = 0.81 # 加速係数
    
    # 最小値，最大値設定
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

    # 各粒子の初期位置, 速度, personal best, global best 設定
    position = []
    velocity = []
    personal_best_scores = []
    # 初期位置, 初期速度
    for s in range(SWARM_SIZE):
        position.append({"x": random.uniform(x_min, x_max), "y": random.uniform(y_min, y_max)})
        velocity.append({"x": 0.0, "y": 0.0})  
    # personal best
    personal_best_positions = list(position)
    for p in position:
        personal_best_scores.append(evaluation(p["x"], p["y"]))
    #global best
    best_particle = np.argmin(personal_best_scores)
    best_position = personal_best_positions[best_particle]

    # ループ回数分Particle移動
    for i in range(ITERATION):
        for s in range(SWARM_SIZE):
            # 変更前の情報の代入
            x, y = position[s]["x"], position[s]["y"]
            vx, vy = velocity[s]["x"], velocity[s]["y"]
            p = personal_best_positions[s]
            
            # 粒子の位置更新
            new_x, new_y = update_position(x, y, vx, vy)
            position[s] = {"x": new_x, "y": new_y}
            # 粒子の速度更新
            new_vx, new_vy = update_velocity(new_x, new_y, vx, vy, p, best_position, W, C1, C2)
            velocity[s] = {"x": new_vx, "y": new_vy}

            # 評価値を求める
            score = evaluation(new_x, new_y)
            # update personal best
            if score < personal_best_scores[s]:
                personal_best_scores[s] = score
                personal_best_positions[s] = {"x": new_x, "y": new_y}
        # update global best
        best_particle = np.argmin(personal_best_scores)
        best_position = personal_best_positions[best_particle]

        # Visualization
        if i == 0 or i == 9 or i == 19 or i == 29:
            visualization(personal_best_positions, SWARM_SIZE)
    
    # Optimal solution
    print("Best Position:", best_position)
    print("Score:", min(personal_best_scores))
    # print("Best Particle:", best_particle)

if __name__ == '__main__':
    main()