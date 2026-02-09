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
#     display_name: Py(Python note)
#     language: python
#     name: pynote
# ---

# # 行列式 determinant

# ## determinant 含义
#
# 行列式的几何意义非常直观,它本质上描述的是**线性变换对空间"体积"的缩放比例**.
#
# 简单来说，你可以把一个矩阵看作是一组基向量的集合，而行列式的绝对值就是这些基向量所张成的平行多面体的体积。
#
# 我们可以从低维到高维来具体理解：
#
# ### 1. 一维空间
# 在一维空间中，矩阵退化成一个数 $a$。
# *   **几何意义**：这个数的绝对值 $|a|$ 代表了数轴上的“长度”缩放比例。
# *   **例子**：如果 $a = 3$，意味着数轴被拉长了 3 倍；如果 $a = -2$，意味着拉长 2 倍并反向。

import numpy as np

# 1d means length
line = np.array([1, 2])
length = np.sqrt(line @ line)
print(length)

# ### 2. 二维空间
# 在二维空间中，矩阵由两个向量组成。
# *   **几何意义**：行列式的绝对值等于这两个向量所构成的**平行四边形的面积**。
# *   **直观理解**：
#     *   假设原来的单位正方形（面积为 1）由基向量 $\vec{i}$ 和 $\vec{j}$ 构成。
#     *   经过矩阵 $A$ 变换后，这两个基向量变成了新的向量。
#     *   这两个新向量围成的平行四边形的面积，就是 $|\det(A)|$。
# *   **符号的意义**：行列式的正负代表了空间的**定向**。
#     *   **正值**：保持定向（例如，原本逆时针排列的向量，变换后仍是逆时针）。
#     *   **负值**：改变定向（例如，图形发生了“翻折”，逆时针变成了顺时针）。

# vector[1,0] 与 vector[1,1] 组成的平行四边形
rect = np.array([[1, 0], [1, 1]])
area = np.linalg.det(rect)
print(area)

# ### 3. 三维空间
#
# 在三维空间中，矩阵由三个向量组成。
# *   **几何意义**：行列式的绝对值等于这三个向量所构成的**平行六面体的体积**。
# *   **符号的意义**：同样代表定向（通常对应“左手系”或“右手系”的变化）。

box = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
volumn = np.linalg.det(box)
print(volumn)

# ### 4. 核心推论：行列式为 0
# 如果行列式的值为 **0**，意味着变换后的“体积”为 0。
# *   **几何解释**：空间被压缩了。在二维中，这意味着整个平面被压缩成了一条线或者一个点；在三维中，意味着被压缩成了一个平面、一条线或一个点。
# *   **数学含义**：这对应着**降维**，说明矩阵的列向量是线性相关的（共线或共面），此时矩阵不可逆（奇异矩阵）。
# *   **行列式的值=0，就是无解**
#
#
# $\begin{vmatrix} 4 & 2 \\2&1\end{vmatrix} = 0$ `[4,2]` and `[2,1]` 在一条线上, 被压缩为一条线了
#
# ### 总结
#
# 行列式不仅仅是一个计算数值，它是一个**缩放因子**：
#
# $$ \text{变换后的有向体积} = \det(A) \times \text{变换前的有向体积} $$
#
# $$ \text{变换后的体积} = \det(A) \times \begin{bmatrix} a & b \\ c & d \end{bmatrix} \times \begin{bmatrix} 1&0&0\\0&1&0\\0&0&1\end{bmatrix} $$
#
# 它告诉你：**经过这个矩阵代表的线性变换后，空间是被拉伸了、压缩了，还是被彻底压扁了（丢失信息）。**
#
# ### 总结比喻
#
# 你可以把行列式看作是线性变换的“体检报告”：
# | 行列式数值 | 体检结论 | 含义 |
# | :--- | :--- | :--- |
# | > 1 | 发福了 | 空间被拉伸，体积变大。 |
# | = 1 | 保持身材 | 空间只发生旋转或切变，体积不变（如旋转矩阵）。 |
# | 介于0和1之间 | 瘦身了 | 空间被轻微压缩。 |
# | = 0 | 彻底坍塌 | 空间维度降低（如平面变直线），不可逆。 |
# | < 0 | 镜像翻转 | 体积发生了缩放，但空间被像镜子一样翻转了。 |
#
# ## 行列式必须是方的
#
# $\begin{vmatrix} 1 & 2 \\ 4 & 5 \end{vmatrix}$
#
# $\begin{vmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{vmatrix}$

# ## 行列式与他的转置相等
#
# 转置后面积不变

# +
import numpy as np

# 1. 创建一个随机的方阵 (例如 4x4)
# 使用随机数可以避免特殊矩阵的巧合，更具普遍性
rng = np.random.default_rng(42)  # 设置随机种子以便结果可复现
A = rng.random((4, 4))

print("原始矩阵 A:")
print(A)

# 2. 计算原始矩阵的行列式
det_A = np.linalg.det(A)

# 3. 计算转置矩阵的行列式
A_T = A.T
det_A_T = np.linalg.det(A_T)

print(f"\n原始矩阵的行列式: {det_A}")
print(f"转置矩阵的行列式: {det_A_T}")

# 4. 验证两者是否相等
# 由于浮点数精度问题，使用 allclose 比直接用 == 更安全
are_equal = np.allclose(det_A, det_A_T)

print(f"\n两者是否相等？ {are_equal}")

# 5. 额外验证：差值是多少（通常是非常接近 0 的极小值）
print(f"差值 (应接近 0): {det_A - det_A_T}")
# -

data = np.arange(1, 5)
data = data.reshape(2, 2)
print(data)
print(np.linalg.det(data))
print(np.linalg.det(data.T))
assert np.allclose(np.linalg.det(data), np.linalg.det(data.T))
