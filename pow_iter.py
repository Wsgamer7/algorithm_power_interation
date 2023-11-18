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


A = np.array([[1, 2], [0, 2]])
B = np.array([[-2, 1, 3], [0, 1, 3], [0, 0, 3]])
C = np.array([[-2, 0, 0], [0, 2, 0], [0, 0, -2]])
print(get_lambda1(C))
