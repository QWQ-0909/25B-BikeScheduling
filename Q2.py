import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, LpInteger, lpSum, LpStatus, PULP_CBC_CMD

# ==================== 1. 数据准备 ====================
parking_spots = ["东门", "南门", "北门", "一食堂", "二食堂", "三食堂", "梅苑1栋", "菊苑1栋",
                 "教学2楼", "教学4楼", "计算机学院", "工程中心", "网球场", "体育馆", "校医院"]

# 需求差数据（正数需拉走，负数需补充）
demand_diff = {
    "7:00": [9, 13, 41, -4, -63, 8, -35, -2, 7, 4, 18, -13, 9, 6, -2],
}

all_nodes = ["运维处"] + parking_spots

# 节点与 ID 的相互映射字典 (解决中文变量名截断 Bug)
N2I = {name: idx for idx, name in enumerate(all_nodes)}
I2N = {idx: name for idx, name in enumerate(all_nodes)}

# 距离矩阵生成
node_coords = {
    "运维处": (50, 5),   
    "东门": (90, 50), "南门": (50, 10), "北门": (50, 90),
    "一食堂": (30, 70), "二食堂": (70, 70), "三食堂": (70, 30),
    "梅苑1栋": (80, 20), "菊苑1栋": (80, 40),
    "教学2楼": (30, 40), "教学4楼": (40, 40), "计算机学院": (20, 60),
    "工程中心": (10, 50), "网球场": (30, 90), "体育馆": (20, 80), "校医院": (90, 70)
}
distance_matrix = {}
for i in all_nodes:
    for j in all_nodes:
        if i == j: distance_matrix[(i, j)] = 0.0
        else:
            dx = node_coords[i][0] - node_coords[j][0]
            dy = node_coords[i][1] - node_coords[j][1]
            distance_matrix[(i, j)] = round(np.sqrt(dx**2 + dy**2) * 0.05, 2)

# ==================== 2. 模型建立 ====================
model = LpProblem("Bike_Sharing_Scheduling", LpMaximize)

K = 3 # 车辆数
vehicles = range(1, K + 1)
time_windows = ["7:00"] 

# 使用 ID (N2I) 来定义变量，避免底层中文截断
x = LpVariable.dicts("Route", [(N2I[i], N2I[j], k, t) for i in all_nodes for j in all_nodes 
                               for k in vehicles for t in time_windows], cat='Binary')

y = LpVariable.dicts("Load", [(N2I[i], k, t) for i in parking_spots 
                               for k in vehicles for t in time_windows], 
                      lowBound=-20, upBound=20, cat=LpInteger)

y_abs = LpVariable.dicts("LoadAbs", [(N2I[i], k, t) for i in parking_spots 
                                      for k in vehicles for t in time_windows], 
                         lowBound=0, upBound=20, cat=LpInteger)

# MTZ 步数变量，消除子回路
u = LpVariable.dicts("Step", [(N2I[i], k, t) for i in parking_spots 
                               for k in vehicles for t in time_windows], 
                      lowBound=1, upBound=len(parking_spots), cat=LpInteger)

# ==================== 3. 目标函数 ====================
# 主目标：最大化满足的需求量；次目标：惩罚路径成本
model += lpSum(y[N2I[i], k, t] * np.sign(demand_diff[t][parking_spots.index(i)]) 
               for i in parking_spots for k in vehicles for t in time_windows) \
         - 0.001 * lpSum(distance_matrix[(i, j)] * x[N2I[i], N2I[j], k, t] 
                         for i in all_nodes for j in all_nodes if i != j for k in vehicles for t in time_windows)

# ==================== 4. 约束条件 ====================
for t in time_windows:
    for k in vehicles:
        for i in parking_spots:
            # 1. 绝对值线性化
            model += y_abs[N2I[i], k, t] >= y[N2I[i], k, t]
            model += y_abs[N2I[i], k, t] >= -y[N2I[i], k, t]
            # 2. 大M约束
            model += y_abs[N2I[i], k, t] <= 20 * lpSum(x[N2I[j], N2I[i], k, t] for j in all_nodes if i != j)

        # 3. 容量约束
        model += lpSum(y_abs[N2I[i], k, t] for i in parking_spots) <= 20

        # 4. 时间窗约束
        model += lpSum(distance_matrix[(i, j)] / 25 * 60 * x[N2I[i], N2I[j], k, t] 
                       for i in all_nodes for j in all_nodes if i != j) <= 30

        # 5. 路径连续性
        for p in all_nodes:
            model += lpSum(x[N2I[i], N2I[p], k, t] for i in all_nodes if i != p) == \
                     lpSum(x[N2I[p], N2I[j], k, t] for j in all_nodes if j != p)

        # 6. 起始点与闲置约束 (允许 <= 1)
        model += lpSum(x[N2I["运维处"], N2I[j], k, t] for j in parking_spots) <= 1
        model += lpSum(x[N2I[j], N2I["运维处"], k, t] for j in parking_spots) <= 1

        # 7. 强制休眠约束 (没离开运维处就不许动)
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    model += x[N2I[i], N2I[j], k, t] <= lpSum(x[N2I["运维处"], N2I[p], k, t] for p in parking_spots)

        # 8. MTZ 消除子回路约束
        N_spots = len(parking_spots)
        for i in parking_spots:
            for j in parking_spots:
                if i != j:
                    model += u[N2I[i], k, t] - u[N2I[j], k, t] + N_spots * x[N2I[i], N2I[j], k, t] <= N_spots - 1

    # 9. 严格供需边界约束
    for i in parking_spots:
        D = demand_diff[t][parking_spots.index(i)]
        if D > 0:
            model += lpSum(y[N2I[i], k, t] for k in vehicles) <= D
            for k in vehicles: model += y[N2I[i], k, t] >= 0
        elif D < 0:
            model += lpSum(y[N2I[i], k, t] for k in vehicles) >= D
            for k in vehicles: model += y[N2I[i], k, t] <= 0
        else:
            for k in vehicles: model += y[N2I[i], k, t] == 0

# ==================== 5. 求解与输出 ====================
# 设置求解器参数：限制 60 秒，或达到 0.1% 的相对容差界限即停止，避免穷举地狱
model.solve(PULP_CBC_CMD(timeLimit=60, gapRel=0.001, msg=1))

print(f"求解状态: {LpStatus[model.status]}")
if model.status == 1:
    for t in time_windows:
        print(f"\n--- 时间窗口: {t} ---")
        for k in vehicles:
            route = []
            curr_id = N2I["运维处"]
            
            # 判断车辆是否出勤
            if sum(x[N2I["运维处"], N2I[j], k, t].varValue for j in parking_spots if x[N2I["运维处"], N2I[j], k, t].varValue) < 0.5:
                print(f"车辆{k}: 闲置待命")
                continue

            # 追踪路线
            for _ in range(len(all_nodes)):
                for j_id in range(len(all_nodes)):
                    if x[curr_id, j_id, k, t].varValue is not None and x[curr_id, j_id, k, t].varValue > 0.9:
                        node_name = I2N[j_id]
                        load = 0
                        if node_name != "运维处":
                            load = y[j_id, k, t].varValue
                        route.append(f"{node_name}({load:+.0f})")
                        curr_id = j_id
                        break
                if curr_id == N2I["运维处"]: break
            
            total_load = sum(y_abs[N2I[i], k, t].varValue for i in parking_spots if y_abs[N2I[i], k, t].varValue)
            print(f"车辆{k}: 运维处 -> {' -> '.join(route)}, 车辆总工作负荷: {total_load} 辆")