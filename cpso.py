# coding: utf-8

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

CPSO = True # CPSOを使うか
AIWF = True # AIWFを使うか

SCORE = 1.000e-15 # CPSOの停止条件SCORE閾値
CPSO_LOOP = 10 # CPSOの停止条件回数
SWARM_SIZE = 100 # 粒子数
ITERATION = 30 # PSOループ回数
C1 = C2 = 2.0 # 加速係数
K = 100 # CLSでの最大反復数
R = 0.25 # 探索範囲縮小関数におけるパラメータ
MIN_X, MIN_Y = -5.0, -5.0 # 探索開始時の範囲最小値
MAX_X, MAX_Y = 5.0, 5.0 # 探索開始時の範囲最大値

# 評価関数(z = x^2+y^2)
def evaluation(x, y):
    return x**2 + y**2 # Sphere function
    #return (1.0 - x)**2 + 100 * (y - x**2)**2 # Rosenbrock function

# PSO
def pso(W, position, velocity, personal_best_scores, personal_best_positions, best_position, search_space_x, search_space_y):
    # ループ回数分Particle移動
    for i in range(ITERATION):
        #AIWF
        if AIWF == True:
            W = aiwf(personal_best_scores, W)

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
        '''
        if i == 0 or i == 1 or i == 2 or i == 3:
            print("ITERATION = " + str(i+1))
            print(W)
            print("")
            visualization(personal_best_positions)
        '''

# Adaptive Inertia Weight Factor (AIWF) 関数
def aiwf(scores, W):
    for s in range(SWARM_SIZE):
        if scores[s] <= np.mean(scores):
            W[s] = (np.min(W) + (((np.max(W) - np.min(W)) * (scores[s] - np.min(scores))) / 
                    (np.mean(scores) - np.min(scores))))
        else:
            W[s] = np.max(W)
    return W

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
    mesh_x = np.arange(-0.5e-3, 0.5e-3, 0.1e-4)
    mesh_y = np.arange(-0.5e-3, 0.5e-3, 0.1e-4)
    mesh_X, mesh_Y = np.meshgrid(mesh_x, mesh_y)
    mesh_Z = evaluation(mesh_X, mesh_Y)
    #ax.plot_wireframe(mesh_X, mesh_Y, mesh_Z)
    # Particle
    for j in range(SWARM_SIZE):
        z = evaluation(positions[j]["x"], positions[j]["y"])
        ax.scatter(positions[j]["x"], positions[j]["y"], z)
    ax.set_xlim([-1.0e-4, 1.0e-4])
    ax.set_ylim([-1.0e-4, 1.0e-4])
    ax.set_zlim([-1.0e-4, 1.0e-4])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.show()

# Chaotic Local Search (CLS)
def cls(top_scores, top_positions):
    cx = []
    cy = []

    min_x, min_y = min(top["x"] for top in top_positions), min(top["y"] for top in top_positions)
    max_x, max_y = max(top["x"] for top in top_positions), max(top["y"] for top in top_positions)
    
    for i in range(len(top_scores)):
        cx.append((top_positions[i]["x"] - min_x) / (max_x - min_x))
        cy.append((top_positions[i]["y"] - min_y) / (max_y - min_y))

    for i in range(K):
        logistic_x = []
        logistic_y = []
        chaotic_scores = []
        chaotic_positions = []
        for j in range(len(top_scores)):
            logistic_x.append(4 * cx[j] * (1 - cx[j]))
            logistic_y.append(4 * cy[j] * (1 - cy[j]))
            # 位置の更新
            chaotic_x, chaotic_y = chaotic_update_position(logistic_x[j], logistic_y[j], min_x, min_y, max_x, max_y)
            chaotic_positions.append({"x": chaotic_x, "y": chaotic_y})
            # score評価
            chaotic_scores.append(evaluation(chaotic_x, chaotic_y))
        # 新しいscoreが前より優れていれば値を返し，それ以外はcx, cyを更新して繰り返す
        if min(chaotic_scores) < min(top_scores):
            print("Better Chaotic Particle found")
            return chaotic_scores, chaotic_positions
        cx = logistic_x
        cy = logistic_y
    return chaotic_scores, chaotic_positions

# Chaotic position更新
def chaotic_update_position(x, y, min_x, min_y, max_x, max_y):
    new_x = min_x + x * (max_x - min_x)
    new_y = min_y + y * (max_y - min_y)
    return new_x, new_y

# 探索範囲縮小
def search_space_reduction(top_positions):
    min_x = []
    min_y = []
    max_x = []
    max_y = []

    min_x.append(min(top["x"] for top in top_positions))
    min_y.append(min(top["y"] for top in top_positions))
    max_x.append(max(top["x"] for top in top_positions))
    max_y.append(max(top["y"] for top in top_positions))
    x = [top.get("x") for top in top_positions]
    y = [top.get("y") for top in top_positions]
    
    for i in range(len(top_positions)):
        min_x.append(x[i] - R * (max_x[0] - min_x[0]))
        min_y.append(y[i] - R * (max_y[0] - min_y[0]))
        max_x.append(x[i] + R * (max_x[0] - min_x[0]))
        max_y.append(y[i] + R * (max_y[0] - min_y[0]))
    
    # 論文通り new_min_x = max(min_x) などにすると最小値と最大値が逆転してしまうので，修正
    new_min_x, new_min_y = min(min_x), min(min_y)
    new_max_x, new_max_y = max(max_x), max(max_y)
    search_space_x = {"min": new_min_x, "max": new_max_x}
    search_space_y = {"min": new_min_y, "max": new_max_y}

    return search_space_x, search_space_y

# Particleを新しい探索範囲内で再生成
def re_generation_particle(W, N, velocity, top_positions, search_space_x, search_space_y):
    new_position = top_positions
    new_velocity = []
    new_W = []
    
    for i in range(int(SWARM_SIZE/5)):
        new_velocity.append(velocity[i])
        new_W.append(W[i])
    
    for n in range(N):
        new_position.append({"x": random.uniform(search_space_x["min"], search_space_x["max"]), "y": random.uniform(search_space_y["min"], search_space_y["max"])})
        new_velocity.append({"x": random.uniform(0, 1), "y": random.uniform(0, 1)})
        new_W.append(random.uniform(0.4, 0.9))
    return new_position, new_velocity, new_W

# 再評価
def re_evaluation(position, personal_best_scores, personal_best_positions, best_position):
    for s in range(SWARM_SIZE):
        x, y = position[s]["x"], position[s]["y"]
        # 評価値を求める
        score = evaluation(x, y)
        # update personal best
        if score < personal_best_scores[s]:
            personal_best_scores[s] = score
            personal_best_positions[s] = {"x": x, "y": y}
    # update global best
    best_particle = np.argmin(personal_best_scores)
    best_position = personal_best_positions[best_particle]
    return personal_best_scores, personal_best_positions, best_position


def run(W, position, velocity, personal_best_scores, personal_best_positions, best_position, search_space_x, search_space_y):

    if CPSO == True:
        i = 0
        while i < CPSO_LOOP:
            print("CPSO: " + str(i+1))
            # PSO
            pso(W, position, velocity, personal_best_scores, personal_best_positions, best_position, search_space_x, search_space_y)

            # 上位1/5を取り出す
            tmp = []
            top_scores = []
            top_positions = []
            tmp = sorted(zip(personal_best_scores, personal_best_positions, velocity, W), key=lambda x: x[0])
            personal_best_scores, personal_best_positions, velocity, W = zip(*tmp)
            personal_best_scores, personal_best_positions, velocity, W = list(personal_best_scores), list(personal_best_positions), list(velocity), list(W)

            for n in range(int(SWARM_SIZE/5)):
                top_scores.append(personal_best_scores[n])
                top_positions.append(personal_best_positions[n])

            # CLS
            print("CLS")
            print("before: " + str(min(top_scores)))
            top_scores, top_positions = cls(top_scores, top_positions)
            print("after: " + str(min(top_scores)))
            #探索範囲縮小
            print("Search Area")
            print("before")
            print("x: " + str(search_space_x))
            print("y: " + str(search_space_y))
            search_space_x, search_space_y = search_space_reduction(top_positions)
            print("after")
            print("x: " + str(search_space_x))
            print("y: " + str(search_space_y))

            # 4/5のParticleを新しい探索範囲内で再生成して追加
            position, velocity, W = re_generation_particle(W, int(SWARM_SIZE - (SWARM_SIZE/5)), velocity, top_positions, search_space_x, search_space_y)
            # 再評価
            personal_best_scores, personal_best_positions, best_position = re_evaluation(position, personal_best_scores, personal_best_positions, best_position)
            # 再評価結果
            print(min(personal_best_scores))
            print("")
            
            # CPSO Visualization
            #if i < 5:
            #    visualization(personal_best_positions)
            
            i += 1
            # Best score が閾値を下回ったら停止
            if min(personal_best_scores) < SCORE:
                return best_position, personal_best_scores
    else:
        # PSO
        pso(W, position, velocity, personal_best_scores, personal_best_positions, best_position, search_space_x, search_space_y)
    return best_position, personal_best_scores


def main():
    W = [] # 慣性係数パラメータ
    if AIWF == True:
        for s in range(SWARM_SIZE):
            W.append(random.uniform(0.4, 0.9))
    else:
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