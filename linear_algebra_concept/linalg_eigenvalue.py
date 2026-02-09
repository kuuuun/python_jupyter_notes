# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # eigenvalue 特征值和特征向量

# 特征值分解（Eigen-decomposition）听起来很吓人，但它的几何意义非常简单：  
# **对于一个矩阵（变换），是否存在一些向量，在变换后方向不变，只是被拉伸或压缩了？**  
# 这些向量就是 **特征向量** ，拉伸的倍数就是 **特征值** 。
#
# 缩放特征值是缩放值，但没有特征向量
#
# ---
# > **相当于在三维空间找到旋转轴，这个轴就是"特征向量"。"特征值“就是这个轴是否被缩放。**
# > **在线性变化中，并不会关心坐标系，特征向量就更有参考意义。**
#
# 公式：
#
# $A\vec{v}=\lambda\vec{v}$
#
# $A\colon$ 特征值 $\vec{v}\colon$ 特征向量 $\lambda\colon$ 特征值
#
# $A\vec{v}-\lambda I\vec{v}=0$
#
# $(A-\lambda I)\vec{v}=0$
#
# $det(A-\lambda I) =0$
#
# ---

# ## example
#
# 已知：$A = \begin{bmatrix} 3 & 1\\0 &2\end{bmatrix}$ 求特征值和特征向量
#
# $det(\begin{bmatrix} 3-\lambda & 1\\0 &2-\lambda\end{bmatrix})=(3-\lambda)(2-\lambda)-1\times0$
#
#
# 特征值 $\lambda=3,\space A\to\begin{bmatrix}0&1\\0&-1\end{bmatrix}$
#
# 特征值 $\lambda=2,\space A\to\begin{bmatrix}1&1\\0&0\end{bmatrix}$

import numpy as np

# +
data = np.array([[3, 1], [0, 2]])

# 创建与data同维度的单位矩阵
I = np.eye(data.shape[0])

eig_values, eig_vectors = np.linalg.eig(data)

# 特征向量 (eigenvectors, 每一列对应一个特征值)
for i in range(len(eig_values)):
    vec = eig_vectors[:, i]  # 取所有行中的第i个元素
    val = eig_values[i]
    print(f"特征值为{val},特征向量为{vec}")
    # 检验公式
    assert np.allclose(data @ vec, val * vec)
    assert np.allclose((data - val * I) @ vec, np.zeros_like(vec))
    assert np.allclose(np.linalg.det(data - val * I), 0)
# -

# ## 求零空间

# +
import numpy as np


# 已知矩阵，和特征值，求特征向量。要用SVD求零空间
# 最简单的办法是用，np.linalg.eig()
def null_space(A, atol=1e-13):
    """
    计算矩阵 A 的零空间（核空间）
    返回一个向量，代表方程 Av=0 的解的方向
    """
    u, s, vh = np.linalg.svd(A)
    # 找出奇异值接近 0 的位置（即那些小于 atol 的值）
    null_mask = s <= atol
    # 对应的 vh 行向量就是基础解系
    null_space_vectors = vh[null_mask].T
    # 如果存在零空间，返回第一个基向量（对于特征向量问题通常够用）
    if null_space_vectors.size > 0:
        return null_space_vectors[:, 0]
    else:
        return np.zeros(A.shape[1])


# --- 示例 ---
# 已知矩阵 A 和特征值 lambda
A = np.array([[3, 1], [0, 2]])
lambda_val = 2  # 假设这是已知的特征值

# 1. 构造矩阵 (A - lambda*I)
I = np.eye(A.shape[0])
matrix = A - lambda_val * I

# 2. 求解零空间（即特征向量）
eigenvector = null_space(matrix)

# 3. 通常我们会将特征向量归一化（长度变为1），上面的函数返回的可能不是单位向量
eigenvector_normalized = eigenvector / np.linalg.norm(eigenvector)

print("特征向量:", eigenvector_normalized)
