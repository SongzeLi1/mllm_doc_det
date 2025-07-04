# 根据Recall和Precision计算F1值
# 计算F1值
def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


def calculate_acc(P, N, precision, recall):
    """
    P: 正样本数量
    N: 负样本数量
    """
    # 计算 TP 和 FN
    TP = recall * P
    FN = P - TP

    # 计算 FP
    if precision == 0:
        FP = float("inf")  # 处理除零错误
    else:
        FP = TP / precision - TP

    # 计算 TN
    TN = N - FP

    # 计算 ACC（确保分母不为零）
    total = P + N
    if total == 0:
        return 0.0
    acc = (TP + TN) / total

    return acc


if __name__ == "__main__":
    # 测试数据
    recall = 0.9403
    precision = 0.6745

    # 计算F1值
    f1 = calculate_f1(precision, recall)
    acc = calculate_acc(729, 341, precision, recall)

    # 输出结果
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Acc: {acc}")
