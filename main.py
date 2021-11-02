import numpy as np
import matplotlib.pyplot as plt

num = 1000


# 生成样本
def creSamples(num_c):
    # 混合系数
    alpha_v = np.array([0.25, 0.25, 0.25, 0.25])
    # alpha_v = np.array([0.1, 0.1, 0.2, 0.4])
    # 按混合系数分配数量
    num_c *= np.ones((4,)) * alpha_v
    # 在 multivariate_normal 方法中 size 参数需要为整数
    num_c = num_c.astype("int32")
    # 设置高斯各混合成分的均值
    mean_c0 = np.array([1, 1])
    mean_c1 = np.array([1, 5])
    mean_c2 = np.array([5, 1])
    mean_c3 = np.array([5, 5])
    # 设置高斯各混合成分的协方差矩阵
    cov0 = np.array([[1, 0], [0, 1]])
    cov1 = np.array([[1, 0], [0, 1]])
    cov2 = np.array([[1, 0], [0, 1]])
    cov3 = np.array([[1, 0], [0, 1]])
    # 按混合系数生成高斯混合分布
    data_c0 = np.random.multivariate_normal(mean_c0, cov0, size=num_c[0])
    data_c1 = np.random.multivariate_normal(mean_c1, cov1, size=num_c[1])
    data_c2 = np.random.multivariate_normal(mean_c2, cov2, size=num_c[2])
    data_c3 = np.random.multivariate_normal(mean_c3, cov3, size=num_c[3])
    ref = np.array([0] * num_c[0] + [1] * num_c[1] + [2] * num_c[2] + [3] * num_c[3]).reshape(np.sum(num_c), 1)
    # 整合生成的数据
    data = np.vstack((data_c0, data_c1, data_c2, data_c3))
    # 按相同顺序打乱样本和其对应的簇标记
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(ref)
    # 画出散点图
    plt.scatter(data_c0[..., 0], data_c0[..., 1], c='pink', marker="o")
    plt.scatter(data_c1[..., 0], data_c1[..., 1], c='greenyellow', marker="^")
    plt.scatter(data_c2[..., 0], data_c2[..., 1], c='lightblue', marker="p")
    plt.scatter(data_c3[..., 0], data_c3[..., 1], c='cyan', marker="D")
    return data, ref


# 得到{d^2}
def dis_squ(data, mu_v):
    # k:簇类数, m:样本数
    k = mu_v.shape[0]
    m = data.shape[0]
    # 生成{d}
    d_squ_mt = np.sum(np.power(data - mu_v[0], 2), axis=1).reshape(m, 1)
    for i in range(1, k):
        d_squ_mt = np.append(d_squ_mt, np.sum(np.power(data - mu_v[i], 2), axis=1).reshape(m, 1), axis=1)
    return d_squ_mt


# 更新均值变量
def mean(data, clu_i):
    k = len(clu_i)
    mean_v = []
    for i in range(k):
        tmp = data[clu_i[i], ...]
        mean_v.append(np.average(tmp, axis=0))
    return mean_v


# k-means
def k_means(data, clu_num):
    # m:样本数
    m = data.shape[0]
    k_index_v = np.random.randint(1, m, clu_num)
    # 设置初始均值向量
    mu_v = data[k_index_v, ...]
    clu_index = []
    update = True
    while update:
        clu_index = []
        d_mt = dis_squ(data, mu_v)
        # 得到簇标记
        lambda_v = np.argmin(d_mt, axis=1).reshape(m, 1)
        for j in range(clu_num):
            clu_index.append(np.where(lambda_v == j)[0])
        mu_new_v = np.array(mean(data, clu_index))
        if np.sum(np.abs(mu_new_v - mu_v)) == 0:
            update = False
        mu_v = mu_new_v
    plt.scatter(mu_v[..., 0], mu_v[..., 1], c="red", marker="2", s=200)
    for i in range(clu_num):
        print(f"{i}: {clu_index[i].size}")
    return clu_index


# 计算高斯分布密度
def GprobDenFunc(data, mu_v, sigma_v):
    m, n = data.shape
    k = mu_v.shape[0]
    p = []
    for i in range(k):
        sigma = np.linalg.det(sigma_v[i])

        coefficient = 1 / (np.power(2 * np.pi, n / 2) * np.sqrt(sigma))
        inv = np.linalg.pinv(sigma_v[i])
        diff_mt = data - mu_v[i]
        row = []
        for j in range(m):
            diff = diff_mt[j].reshape(1, n)
            index = diff @ inv @ diff.T * (-1 / 2)
            row.append(np.linalg.det(index))
        p.append(coefficient * np.exp(np.array(row)))
    return np.array(p)


def posterioriProb(alpha_v, p_mt):
    numerator = p_mt * alpha_v
    denominator = (p_mt.T @ alpha_v).T
    gamma_mt = numerator / denominator
    return gamma_mt.T


def expectationMaximization(data, clu_num, times):
    # m:样本数, n:特征数, k:簇类数
    m, n = data.shape
    k = clu_num
    # 设置初始模型参数
    alpha_v = 1 / k * np.ones((k, 1))
    k_index_v = np.random.randint(1, m, clu_num)
    mu_v = data[k_index_v, ...]
    sigma_v = np.array((0.1 * np.identity(n)).tolist() * k).reshape(k, n, n)
    gamma_mt = np.ones((m, k))
    learn_times = 0
    while learn_times < times:
        learn_times += 1

        p_mt = GprobDenFunc(data, mu_v, sigma_v)
        gamma_mt = posterioriProb(alpha_v, p_mt)

        mu_new_v = []
        sigma_new_v = []
        alpha_new_v = []
        for i in range(k):
            denominator = np.sum(gamma_mt[..., i])
            # mu
            numerator = data.T @ gamma_mt[..., i]
            mu_new_v.append(numerator / denominator)
            # sigma
            sigma_mt = np.zeros((n, n))
            gamma_i = gamma_mt[..., i].reshape(m, 1)
            for j in range(m):
                diff = (data[j] - mu_v[i]).reshape(1, n)
                sigma_mt += gamma_i[j] * (diff.T @ diff)
            sigma_new_v.append(sigma_mt / denominator)
            # alpha
            alpha_new_v.append(denominator / m)
        mu_v = np.array(mu_new_v).reshape(k, n)
        sigma_v = np.array(sigma_new_v)
        alpha_v = np.array(alpha_new_v).reshape(k, 1)
    lambda_v = np.argmax(gamma_mt, axis=1).reshape(m, 1)
    clu_index = []
    for i in range(k):
        clu_index.append(np.where(lambda_v == i)[0])
    for i in range(k):
        print(f"{i}: {clu_index[i].size}")
    return clu_index


def draw(data, clu_i, external):
    print(f"JC: {external[0]}\nFMI: {external[1]}\nRI: {external[2]}")
    plt.scatter(data[clu_i[0], ...][..., 0], data[clu_i[0], ...][..., 1], c="black", marker="+")
    plt.scatter(data[clu_i[1], ...][..., 0], data[clu_i[1], ...][..., 1], c="peru", marker="x")
    plt.scatter(data[clu_i[2], ...][..., 0], data[clu_i[2], ...][..., 1], c="dodgerblue", marker="_")
    plt.scatter(data[clu_i[3], ...][..., 0], data[clu_i[3], ...][..., 1], c="darkviolet", marker="|")


def externalIndex(ref, clu_i):
    m = ref.shape[0]
    k = len(clu_i)
    # 生成观测簇标记
    obs = np.zeros(m)
    for i in range(k):
        for ele in clu_i[i]:
            obs[ele] = i
    obs = obs.reshape(m, 1).astype("int32")
    # 采用外部指标进行性能度量
    a, b, c, d = 0, 0, 0, 0
    for i in range(m):
        for j in range(i):
            if ref[i] == ref[j]:
                if obs[i] == obs[j]:
                    a += 1
                else:
                    c += 1
            else:
                if obs[i] == obs[j]:
                    b += 1
                else:
                    d += 1
    jaccard = a / (a + b + c)
    fm = a / np.sqrt((a + b) * (a + c))
    ri = 2 * (a + d) / (m * (m - 1))
    return jaccard, fm, ri


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_sam, ref_sam = creSamples(num)

    clu_index_sam = k_means(data_sam, 4)
    draw(data_sam, clu_index_sam, externalIndex(ref_sam, clu_index_sam))

    # clu_index_sam = expectationMaximization(data_sam, 4, 100)
    # draw(data_sam, clu_index_sam, externalIndex(ref_sam, clu_index_sam))

    plt.show()
