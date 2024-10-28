import numpy as np

# 极小型指标的正向化
def min_to_max(matrix, column):
    min_value = matrix[:, column].min()
    if min_value == 0:
        min_value = 1e-10  # 避免除以0
    matrix[:, column] = min_value / matrix[:, column]
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




def calculate_entropy(matrix):
    epsilon = 1e-10  # 防止对0取对数
    norm_matrix = matrix / matrix.sum(axis=0)
    # 检查分母是否为零或非常接近零
    if matrix.shape[0] <= 1:
        raise ValueError("矩阵的行数必须大于1，以便正确计算熵。")

    entropy = -np.sum(norm_matrix * np.log(norm_matrix + epsilon), axis=0) / np.log(matrix.shape[0])
    return entropy

def entropy_weight(matrix):
    entropy = calculate_entropy(matrix)
    weights = (1 - entropy) / (1 - entropy).sum()
    return weights

def topsis(matrix, types):
    print("原始矩阵：\n", matrix)

    for i, typ in enumerate(types):
        if typ == 'min':
            matrix = min_to_max(matrix, i)
        elif typ == 'mid':
            best_value = np.median(matrix[:, i])
            matrix = mid_to_max(matrix, i, best_value)
        elif typ == 'interval':
            q1, q3 = np.percentile(matrix[:, i], [25, 75])
            matrix = interval_to_max(matrix, i, q1, q3)

    print("\n正向化后的矩阵：\n", matrix)

    norm_matrix = normalize_matrix(matrix)
    print("\n标准化后的矩阵：\n", norm_matrix)

    weights = entropy_weight(norm_matrix)
    print("\n权重：\n", weights)

    weighted_matrix = weighted_normalization(norm_matrix, weights)
    print("\n加权标准化矩阵：\n", weighted_matrix)

    ideal_solution, negative_ideal_solution = ideal_solutions(weighted_matrix)

    distance_to_ideal, distance_to_negative_ideal = calculate_distances(weighted_matrix, ideal_solution, negative_ideal_solution)

    closeness = relative_closeness(distance_to_ideal, distance_to_negative_ideal)
    ranking = np.argsort(closeness)[::-1] + 1

    scores = closeness / np.sum(closeness)

    return weights, closeness, ranking, scores


data = np.loadtxt("./DataSet/西瓜数据集.txt",encoding='utf-8', skiprows=1, usecols=range(1, 7))

# 指标类型与权重
types = ['max', 'mid', 'max', 'min', 'max', 'min']

weights, closeness, ranking, scores = topsis(data, types)

print("\n权重：", weights)
print("相对接近度：", closeness[:5])
print("归一化得分：", scores[:5])
print("排序：", ranking)
