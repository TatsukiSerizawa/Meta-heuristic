import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

SWARM_SIZE = 100 # 粒子数
ITERATION = 30 # PSOループ回数
C1 = C2 = 2.0 # 加速係数
MIN_X, MIN_Y = -5.0, -5.0 # 探索開始時の範囲最小値
MAX_X, MAX_Y = 5.0, 5.0 # 探索開始時の範囲最大値

# 評価関数(z = x^2+y^2)
def evaluation(x, y):
    return x**2 + y**2 # Sphere function

# PSO
def pso(W, position, velocity, personal_best_scores, personal_best_positions, best_position, search_space_x, search_space_y):
    # ループ回数分Particle移動
    for i in range(ITERATION):
        for s in range(SWARM_SIZE):
            # 変更前の情報の代入
            x, y = position[s]["x"], position[s]["y"]
            vx, vy = velocity[s]["x"], velocity[s]["y"]
            p = personal_best_positions[s]
            
            # 粒子の位置更新
            new_x, new_y = update_position(x, y, vx, vy, search_space_x, search_space_y)
            position[s] = {"x": new_x, "y": new_y}
            # 粒子の速度更新
            new_vx, new_vy = update_velocity(new_x, new_y, vx, vy, p, best_position, W[s])
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

        # PSO Visualization
        
        if i == 0 or i == 1 or i == 2 or i == 3:
            print("ITERATION = " + str(i+1))
            print(W)
            print("")
            visualization(personal_best_positions)
        

# 粒子の位置更新関数
def update_position(x, y, vx, vy, search_space_x, search_space_y):
    new_x = x + vx
    new_y = y + vy
    # 探索範囲内か確認
    if new_x < search_space_x["min"] or new_y < search_space_y["min"]:
        new_x, new_y = search_space_x["min"], search_space_y["min"]
    elif new_x > search_space_x["max"] or new_y > search_space_y["max"]:
        new_x, new_y = search_space_x["max"], search_space_y["max"]
    return new_x, new_y

# 粒子の速度更新関数
def update_velocity(x, y, vx, vy, p, g, w):
    #random parameter (0~1)
    r1 = random.random()
    r2 = random.random()
    # 速度更新
    new_vx = w * vx + C1 * (g["x"] - x) * r1 + C2 * (p["x"] - x) * r2
    new_vy = w * vy + C1 * (g["y"] - y) * r1 + C2 * (p["y"] - y) * r2
    return new_vx, new_vy

# 可視化関数
def visualization(positions):
    fig = plt.figure()
    ax = Axes3D(fig)
    # Mesh
    #mesh_x = np.arange(-0.5e-3, 0.5e-3, 0.1e-4)
    #mesh_y = np.arange(-0.5e-3, 0.5e-3, 0.1e-4)
    mesh_x = np.arange(-1.0, 1.0, 0.1)
    mesh_y = np.arange(-1.0, 1.0, 0.1)
    mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)
    mesh_Z = evaluation(mesh_X, mesh_Y)
    #ax.plot_wireframe(mesh_X, mesh_Y, mesh_Z)
    # Particle
    for j in range(SWARM_SIZE):
        z = evaluation(positions[j]["x"], positions[j]["y"])
        ax.scatter(positions[j]["x"], positions[j]["y"], z)
    #ax.set_xlim([-1.0e-4, 1.0e-4])
    #ax.set_ylim([-1.0e-4, 1.0e-4])
    #ax.set_zlim([-1.0e-4, 1.0e-4])
    ax.set_xlim([-5.0, 5.0])
    ax.set_ylim([-5.0, 5.0])
    ax.set_zlim([-1.0, 30.0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.show()

def run(W, position, velocity, personal_best_scores, personal_best_positions, best_position, search_space_x, search_space_y):
    pso(W, position, velocity, personal_best_scores, personal_best_positions, best_position, search_space_x, search_space_y)
    return best_position, personal_best_scores

def main():
    W = [] # 慣性係数パラメータ
    for s in range(SWARM_SIZE):
        W.append(0.9)
    

    # 時間計測開始
    start = time.time()

    # 各粒子の初期位置, 速度, personal best, global best 及びsearch space設定
    position = []
    velocity = []
    personal_best_scores = []
    # 初期位置, 初期速度
    for s in range(SWARM_SIZE):
        position.append({"x": random.uniform(MIN_X, MAX_X), "y": random.uniform(MIN_Y, MAX_Y)})
        velocity.append({"x": random.uniform(0, 1), "y": random.uniform(0, 1)})
    # personal best
    personal_best_positions = list(position)
    for p in position:
        personal_best_scores.append(evaluation(p["x"], p["y"]))
    # global best
    best_particle = np.argmin(personal_best_scores)
    best_position = personal_best_positions[best_particle]
    # search space
    search_space_x, search_space_y = {'min': MIN_X, "max": MAX_X}, {"min": MIN_Y, "max": MAX_Y}

    # run
    best_position, personal_best_scores = run(W, position, velocity, personal_best_scores, personal_best_positions, best_position, search_space_x, search_space_y)


    # 時間計測終了
    process_time = time.time() - start

    # Optimal solution
    print("Best Position:", best_position)
    print("Score:", min(personal_best_scores))
    print("time:", process_time)

if __name__ == '__main__':
    main()