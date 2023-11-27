import numpy as np


def get_lambda1(matrix, delta=1e-10, max_iter=1000):
    matrix_rows = np.shape(matrix)[0]
    normalized_vec = np.ones(shape=(matrix_rows))  # 初始向量v0
    max_vec_old = 1

    for i in range(1, max_iter + 1):
        vec = np.matmul(matrix, normalized_vec)
        max_vec = np.max(vec)
        normalized_vec = vec / max_vec
        delta_now = abs(max_vec - max_vec_old)
        if delta_now < delta:
            print("success in the {0}th iter".format(i))
            print("lamda1:{0}, x1:{1}".format(max_vec, normalized_vec))
            return (max_vec, normalized_vec)
        print("the {0}th iter, lamda1:{1}, x1:{2}".format(i, max_vec, normalized_vec))
        max_vec_old = max_vec
    return (False, False)


def get_ture_lambda1(matrix, delta=1e-10, max_iter=1000):
    lambda1, x1 = get_lambda1(matrix, delta, max_iter)
    if lambda1 == False and x1 == False:
        print("可能 lambda1 = - lambda2, 继续迭代")
        lambda1_square, x1 = get_lambda1(np.matmul(matrix, matrix), delta, max_iter)
        if lambda1_square:
            lambda1 = np.sqrt(lambda1_square)
    return lambda1, x1


# 算所有特征值和特征向量
def get_all_lambda(matrix, delta=1e-10, max_iter=1000):
    lambda_x_lst = []
    for i in range(np.shape(matrix)[0]):
        lambda1, x1 = get_ture_lambda1(matrix, delta, max_iter)
        lambda_x_lst.append((lambda1, x1))
        normalized_x1 = x1 / np.linalg.norm(x1)
        matrix = matrix - lambda1 * np.outer(normalized_x1, normalized_x1)
    print("=======================")
    print("特征值，  特征向量")
    for lam, x1 in lambda_x_lst:
        print(
            "{0:.5f}, {1}".format(
                lam,
                x1,
            )
        )
    return lambda_x_lst


# 用法：改A，点运行， A需要是对称阵
A = np.array([[1, 0, 0], [0, 20, 0], [0, 0, -20]], dtype=np.double)
get_all_lambda(A, delta=1e-15)
# A1 = np.array([[-4.0, -10.0, -10.0], [-10.0, -2.0, -10.0], [-10.0, -10.0, 0.0]])
# lambda1_A1 = get_ture_lambda1(A1)
# true_lambda1_A1 = np.linalg.eigvals(A1)
