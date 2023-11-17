import numpy as np


# return(特征值, 特征向量) of 绝对值最大的特征值
def get_max_lambda(mat, delta=1e-10, max_iter=1000):
    iters_count = 1
    N = np.shape(mat)[0]
    u = np.ones(shape=(N))  # init u=v0
    m_old = 1
    delta_now = 1

    while iters_count <= max_iter and delta_now > delta:
        v = np.matmul(mat, u)
        m = np.max(v)  # m=hat_lambda1， 特征值预测值
        u = v / m  # u=hat_x1，特征向量预测值
        delta_now = abs(m - m_old)
        print("第{0}次迭代, lambda1 = {1}, x1 = {2}".format(iters_count, m, u))
        m_old = m
        iters_count += 1

    if iters_count > max_iter and delta_now > delta:
        print("迭代次数达到最大值，迭代结束")
    else:
        print(
            "成功：第{iters_count}次迭代 , lambda1 = {m}".format(
                iters_count=iters_count - 1, m=m
            )
        )


A = np.array([[1, 2], [0, 2]])
B = np.array([[-2, 1, 3], [0, 1, 3], [0, 0, 3]])
get_max_lambda(B, max_iter=1000)
