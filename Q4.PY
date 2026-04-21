import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# ================== 输入数据 ==================
# 优化后的停车点位名称及故障数（示例）
locations = [
    '运维处', '法学院', '校主楼', '材料化工学院', '教学4楼',
    '体育馆', '计算机学院', '工程中心', '校医院', '网球场'
]
demands = [0, 15, 18, 12, 30, 9, 21, 28, 11, 8]   # 故障数（辆），运维处为0

# 距离矩阵（km），需根据实际地图数据填充（此处仅示例前两行）
distance_matrix = np.array([
    [0, 0.5, 1.0, 1.2, 2.0, 0.8, 1.5, 1.8, 2.2, 0.7],   # 运维处到各点
    [0.5, 0, 0.8, 1.0, 1.8, 1.2, 1.0, 1.5, 2.0, 0.9],   # 法学院到各点
    # ... 其他点位的距离数据需补充完整
])

# ================== 模型参数 ==================
vehicle_capacity = 20          # 单次最大运输量
speed = 25                     # 车速 km/h
service_time_per_bike = 1      # 装卸时间（分钟/辆）

# ================== 创建数据模型 ==================
def create_data_model():
    data = {}
    data['distance_matrix'] = distance_matrix.tolist()
    data['demands'] = demands
    data['vehicle_capacity'] = vehicle_capacity
    data['num_vehicles'] = 3        # 鲁迪的车辆数（假设可多次出发）
    data['depot'] = 0               # 运维处索引
    return data

# ================== 路径优化求解 ==================
def optimize_routes(data):
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot']
    )
    routing = pywrapcp.RoutingModel(manager)

    # 定义距离回调函数
    def distance_callback(from_index, to_index):
        return data['distance_matrix'][from_index][to_index]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 添加容量约束
    def demand_callback(from_index):
        return data['demands'][from_index]
    demand_callback_index = routing.RegisterUnaryCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null slack
        [data['vehicle_capacity']] * data['num_vehicles'],
        True,
        'Capacity'
    )