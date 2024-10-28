import numpy as np

# 极小型指标的正向化
def min_to_max(matrix, column):
    max_value = matrix[:, column].max()

    matrix[:, column] = max_value - matrix[:, column]
    return matrix

# 中间型指标的正向化
def mid_to_max(matrix, column, best_value):
    max_value = matrix[:, column].max()
    min_value = matrix[:, column].min()
    if max_value == min_value:
        matrix[:, column] = 1  # 如果最大值和最小值相同，所有值视为最优
    else:
        matrix[:, column] = 1 - abs(matrix[:, column] - best_value) / (max_value - min_value)
    return matrix

# 区间型指标的正向化
def interval_to_max(matrix, column, low, high):
    max_value = matrix[:, column].max()
    min_value = matrix[:, column].min()
    if max_value == min_value:
        matrix[:, column] = 1  # 如果最大值和最小值相同，所有值视为最优
    else:
        matrix[:, column] = np.where((matrix[:, column] >= low) & (matrix[:, column] <= high),
                                     1,
                                     1 - abs(matrix[:, column] - (low + high) / 2) / (max_value - min_value))
    return matrix

# 矩阵标准化，加入防止除以0的保护
def normalize_matrix(matrix):
    norm_factor = np.sqrt((matrix ** 2).sum(axis=0)) + 1e-10  # 防止除以0
    norm_matrix = matrix / norm_factor
    return norm_matrix

# 加权规范化矩阵
def weighted_normalization(matrix, weights):
    return matrix * weights

# 计算理想解和负理想解
def ideal_solutions(weighted_matrix):
    ideal_solution = weighted_matrix.max(axis=0)
    negative_ideal_solution = weighted_matrix.min(axis=0)
    return ideal_solution, negative_ideal_solution

# 计算方案到理想解和负理想解的距离
def calculate_distances(weighted_matrix, ideal_solution, negative_ideal_solution):
    distance_to_ideal = np.sqrt(((weighted_matrix - ideal_solution) ** 2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_matrix - negative_ideal_solution) ** 2).sum(axis=1))
    return distance_to_ideal, distance_to_negative_ideal

# 计算相对接近度
def relative_closeness(distance_to_ideal, distance_to_negative_ideal):
    return distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal + 1e-10)  # 防止除以0

# TOPSIS方法实现
def topsis(matrix, weights, types):
    print("原始矩阵：\n", matrix)

    # 正向化步骤，根据types进行相应的转换
    for i, typ in enumerate(types):
        if typ == 'min':  # 极小型转极大型
            matrix = min_to_max(matrix, i)
        elif typ == 'mid':  # 中间型转极大型，假设理想值为中位数
            best_value = np.median(matrix[:, i])
            matrix = mid_to_max(matrix, i, best_value)
        elif typ == 'interval':  # 区间型转极大型，假设理想区间为[min, max]的前四分位数和后四分位数
            q1, q3 = np.percentile(matrix[:, i], [25, 75])
            matrix = interval_to_max(matrix, i, q1, q3)

    print("\n正向化后的矩阵：\n", matrix)

    # 矩阵标准化
    norm_matrix = normalize_matrix(matrix)
    print("\n标准化后的矩阵：\n", norm_matrix)

    # 加权规范化矩阵
    weighted_matrix = weighted_normalization(norm_matrix, weights)
    print("\n加权标准化矩阵：\n", weighted_matrix)

    # 计算理想解和负理想解
    ideal_solution, negative_ideal_solution = ideal_solutions(weighted_matrix)

    # 计算各方案与理想解和负理想解的距离
    distance_to_ideal, distance_to_negative_ideal = calculate_distances(weighted_matrix, ideal_solution,
                                                                        negative_ideal_solution)

    # 计算相对接近度，并返回排序结果
    closeness = relative_closeness(distance_to_ideal, distance_to_negative_ideal)
    ranking = np.argsort(closeness)[::-1] + 1  # 从1开始排序
    return closeness, ranking


data = np.loadtxt("./DataSet/西瓜数据集.txt",encoding='utf-8', skiprows=1, usecols=range(1, 7))

# 指标类型与权重
weights = np.array([0.15, 0.2, 0.1, 0.1, 0.25, 0.2])
types = ['max', 'mid', 'max', 'min', 'max', 'min']

# 运行 TOPSIS
closeness, ranking = topsis(data, weights, types)
print("\n相对接近度：\n", closeness)
print("\n排序：\n", ranking)
