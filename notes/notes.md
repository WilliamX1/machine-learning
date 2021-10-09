# 机器学习课程笔记

## 线性回归 Linear Regression

### 概念
线性回归就是学习到一个数据属性线性组合的函数来进行未来的预测。

### 基本形式
$$
f(x) = w_0x_0 + w_1x_1 + w_2x_2 + ... + w_nx_n \quad (x_0 === 1)
$$
**向量形式**
$$
f(\vec{x}) = \vec{w}^T\vec{x}
$$

### 假设 Hypothesis
$$
h_{\theta}(x^{i}) = \theta_0 + \theta_{1}x^i \\
$$

### 代价函数 Cost Function
代价函数越大，代表预测值与真实值之间误差越大。所以，我们要尝试不同 $\theta$ 最小化代价函数。

**均方误差代价函数 Mean Squared Error**
$$
J^i(\theta) = \frac{1}{2}(y_i - f_{\theta}(x_i))^2 \\
J(\vec{\theta}) = \frac{1}{2n}\sum_{i = 1}^{m}(h_{\theta}(\vec{x}^i) - \vec{y}^i)^2
$$

### 求解参数 $\theta$ 
#### 解析解 closed-form
$$
\vec{\theta} = (\vec{X}^T\vec{X})^{-1}\vec{X}^Ty
$$
能够一次性求解，但算法的时间复杂度是指数级别的，且会占用大量内存资源。

#### 全量梯度下降法 Batch Gradient Descent （BGD)
每一次梯度下降都利用全部的数据。当数据量 $n$ 很大时，速度很慢。
对代价函数求偏导数
$$
\frac{\partial}{\partial\vec\theta}J(\vec\theta) = \frac{1}{n}\sum_{i = 1}^n(h_{\vec\theta}(\vec{x}^i) - \vec{y}^i)\vec{x_i} \qquad \underset{\theta}{min} \; J(\theta)
$$
求取 $\vec\theta$ 新值
$$
\vec\theta_{new} = \vec\theta_{old} + \eta\frac{1}{n}\sum_{i = 1}^n(y_i - f_\vec\theta(\vec{x_i}))\vec{x_i}
$$

#### 随机梯度下降法 Stochastic Gradient Descent (SGD)
每次梯度下降使用一个随机的样本，相比于 BGD 更快。但如果每次选取的学习率太小则很慢，太大则无法得到全局最优解，所以应该让学习率一开始比较大，随后逐渐减小。这种方法也称作_模拟退火(simulated annealing)_
对代价函数求偏导数
$$
\frac{\partial J^i(\theta)}{\partial\theta} = -(y_i - f_\theta(\vec{x_i}))\vec{x_i}
$$
求取 $\vec\theta$ 新值
$$
\vec\theta_{new} = \vec\theta_{old} + \eta(y_i - f_\theta(x_i))x_i
$$

### 部分梯度下降法 Mini-Batch Gradient Descent
每次梯度下降过程使用一个比较小的随机数据集，能够利用计算机对于矩阵求解的性能优化，从而加快计算效率。 

### 流程总结
1. 初始化一个线性模型。例如 $y = 2 + 3x$ ，则初始参数 $\theta_0 = 2 \theta_1 = 3$
2. 给定一个样本对，例如 $(2, 4)$ ，代入模型中求得预测值，即 $y = 2 + 3 * 2 = 8$
3. 代入代价函数公式求得代价值，即 $J = 1/2 \times (8 - 4)^2$ 。
4. 代入偏导数公式中求 $\theta_0$ 和 $\theta_1$ 的偏导数，即 $\frac{\partial}{\partial \vec\theta} J(\vec\theta) = [2 + 3  \times 2 - 4, (2 + 3 \times 2 - 4) \times 2] = [4, 8]$ 。
5. 假设我们的学习率是 0.1 ，那么代入梯度下降公式得到 $\vec\theta_0 := \vec\theta_0 - \eta\frac{\partial}{\partial\vec \theta} J\vec(\theta) = [2 - 0.1 \times 4, 3 - 0.1 \times 8] = [1.6, 2.2]$
5. 得到新参数，获得新模型 $y = 1.6 + 2.2x$ 。
6. 重复上述 2 ~ 5 过程。

### 相关内容
* 对数几率回归
$$
y = \frac{1}{1 + e^{-z}} \\
ln \frac{y}{1 - y} = \vec{w}^T\vec{x} + b
$$
* 线性判别分析 Linear Discrimination Analysis (LDA)
试图将数据点投影到一条直线上，使得同类数据的投影点尽可能接近，不同类数据的投影点尽可能远离。在对新数据进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定新数据的类别

### 参考
https://zhuanlan.zhihu.com/p/45023349
https://blog.csdn.net/weixin_42546496/article/details/88115095