# coding: utf-8

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

AIWF = True # AIWFを使うか


# 評価関数(z = x^2+y^2)
def evaluation(x, y):
    return x**2 + y**2 # Sphere function
    #return (1.0 - x)**2 + 100 * (y - x**2)**2

# PSO
def pso(ITERATION, SWARM_SIZE, W, C1, C2, position, velocity, personal_best_scores, personal_best_positions, best_position):
    # ループ回数分Particle移動
    for i in range(ITERATION):
        #AIWF
        if AIWF == True:
            W = aiwf(personal_best_scores, W, SWARM_SIZE)

        for s in range(SWARM_SIZE):
            # 変更前の情報の代入
            x, y = position[s]["x"], position[s]["y"]
            vx, vy = velocity[s]["x"], velocity[s]["y"]
            p = personal_best_positions[s]
            
            # 粒子の位置更新
            new_x, new_y = update_position(x, y, vx, vy)
            position[s] = {"x": new_x, "y": new_y}
            # 粒子の速度更新
            if AIWF == True:
                new_vx, new_vy = update_velocity(new_x, new_y, vx, vy, p, best_position, W[s], C1, C2)
            else:
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
        '''
        if i == 0 or i == 1 or i == 2 or i == 3:
            print("ITERATION = " + str(i+1))
            print(W)
            print("")
            visualization(personal_best_positions, SWARM_SIZE)
        '''

# AIWF関数
def aiwf(scores, W, SWARM_SIZE):
    for s in range(SWARM_SIZE):
        if scores[s] <= np.mean(scores):
            W[s] = (np.min(W) + (((np.max(W) - np.min(W)) * (scores[s] - np.min(scores))) / 
                    (np.mean(scores) - np.min(scores))))
        else:
            W[s] = np.max(W)
    return W

# 粒子の位置更新関数
def update_position(x, y, vx, vy):
    new_x = x + vx
    new_y = y + vy
    return new_x, new_y

# 粒子の速度更新関数
def update_velocity(x, y, vx, vy, p, g, w, c1, c2):
    #random parameter (0~1)
    r1 = random.random()
    r2 = random.random()
    # 速度更新
    new_vx = w * vx + c1 * (g["x"] - x) * r1 + c2 * (p["x"] - x) * r2
    new_vy = w * vy + c1 * (g["y"] - y) * r1 + c2 * (p["y"] - y) * r2
    return new_vx, new_vy

# 可視化関数
def visualization(positions, SWARM_SIZE):
    fig = plt.figure()
    ax = Axes3D(fig)
    # Mesh
    mesh_x = np.arange(-3.0, 3.0, 0.1)
    mesh_y = np.arange(-3.0, 3.0, 0.1)
    mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)
    mesh_Z = evaluation(mesh_X, mesh_Y)
    ax.plot_wireframe(mesh_X, mesh_Y, mesh_Z)
    # Particle
    for j in range(SWARM_SIZE):
        z = evaluation(positions[j]["x"], positions[j]["y"])
        ax.scatter(positions[j]["x"], positions[j]["y"], z)
    ax.set_xlim([-5.0, 5.0])
    ax.set_ylim([-5.0, 5.0])
    ax.set_zlim([-2.0, 15.0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.show()


def main():
    SWARM_SIZE = 100 # 粒子数
    ITERATION = 30 # ループ回数
    C1 = 1.49 # 加速係数
    C2 = 1.49 # 加速係数
    if AIWF == True:
        W = [] # 慣性係数パラメータ
        for s in range(SWARM_SIZE):
            W.append(random.uniform(0.0, 1.0))
    else:
        W = 0.9
    
    # 時間計測開始
    start = time.time()
    
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

    # PSO
    pso(ITERATION, SWARM_SIZE, W, C1, C2, position, velocity, personal_best_scores, personal_best_positions, best_position)

    # 上位N件を取り出す
    

    '''
    # ループ回数分Particle移動
    for i in range(ITERATION):
        #AIWF
        if AIWF == True:
            W = aiwf(personal_best_scores, W, SWARM_SIZE)

        for s in range(SWARM_SIZE):
            # 変更前の情報の代入
            x, y = position[s]["x"], position[s]["y"]
            vx, vy = velocity[s]["x"], velocity[s]["y"]
            p = personal_best_positions[s]
            
            # 粒子の位置更新
            new_x, new_y = update_position(x, y, vx, vy)
            position[s] = {"x": new_x, "y": new_y}
            # 粒子の速度更新
            if AIWF == True:
                new_vx, new_vy = update_velocity(new_x, new_y, vx, vy, p, best_position, W[s], C1, C2)
            else:
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
        #if i == 0 or i == 1 or i == 2 or i == 3:
        #    print("ITERATION = " + str(i+1))
        #    print(W)
        #    print("")
        #    visualization(personal_best_positions, SWARM_SIZE)
    '''

    # 時間計測終了
    process_time = time.time() - start

    # Optimal solution
    print("Best Position:", best_position)
    print("Score:", min(personal_best_scores))
    print("time:", process_time)
    # print("Best Particle:", best_particle)

if __name__ == '__main__':
    main()