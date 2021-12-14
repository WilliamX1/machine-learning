# 机器学习课程笔记

## 线性回归 Linear Regression

### <id=1>概念</id>
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

------

## 决策树
### 基本策略
一颗决策树包含一个根结点、若干个内部结点和若干个叶结点。叶结点对应于决策结果，其他每个结点则对应于一个属性测试。每个结点包含的样本集合根据属性测试的结果被划分到子节点中。根节点包含样本全集。采用 _分而治之_ 策略。

### 基本定义
* 信息熵 - Information entropy：信息熵越小，纯度越高。
$$
Ent(D) = -\sum^{|\gamma|}_{k = 1}p_klog_2p_k
$$

* 信息增益 - Information gain：信息增益越大，则意味着使用属性 a 来进行划分所获得的的纯度提升越大。
**ID3 决策树学习算法以此作为准则来选择划分属性**
$$
Gain(D, a) = Ent(D) - \sum_{v = 1}^{V}Ent(D^v)
$$

* 固有值 - Intrinsic value 
$$
IV(a) = -\sum_{v = 1}^V\frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|}
$$
* 增益率 - Gain ratio
**C4.5 算法先从候选划分属性中找出信息增益高于平均水平的属性，再从中选择增益率高的**
$$
Gain\_ratio(D, a) = \frac{Gain(D, a)}{IV(a)}
$$
* 基尼指数 - Gini index：基尼指数越小，数据集 D 的纯度越高。
**CART 决策树使用基尼指数来选择划分属性**
$$
Gini(D) = 1 - \sum_{k = 1}^{|\gamma|}p_k^2
$$
### 剪枝 Pruning
解决_过拟合_的主要手段。

* 预剪枝 - Prepruning
预剪枝是在决策树生成过程中，对每个结点划分前估计，若当前结点划分不能带来决策树泛化性能提升，则停止划分并将当前节点标记为叶结点。
可以降低_过拟合_的风险，显著减少决策树的训练时间开销和测试时间开销。
基于贪心本质，有_欠拟合_风险。

* 后剪枝 - Postpruning
后剪枝是在训练处完整的决策树后自底向上对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点。
_训练开销时间_极大。

### 连续与缺失值
* 连续问题
采用_二分法_对连续属性进行处理。连续属性可以作为后代结点的划分属性。

* 缺失值处理
让样本以不同的概率划入到不同的子节点中。即在信息熵增益时乘以参与计算的数据的概率。

------

## 贝叶斯分类
### 贝叶斯定理 Bayes Rule
$$
P(c | \pmb{x}) = \frac{P(c) P(\pmb{x} | c)}{P(\pmb{x})}
$$

$P(c)$ 是类“先验”概率。

$P(\pmb{x} | c)$ 是样本 $\pmb{x}$ 相对类标记 $c$ 的类条件概率，或称为似然概率。

$P(\pmb{x})$ 是用于归一化的“证据”因子。

### 极大似然估计 Maximum Likelihood Estimation

$D_c$ 表示训练集 D 中第 c 类样本组成的集合，假设这些样本是独立同分布的，则参数 $\theta_{c}$ 对于数据集 $D_c$ 的似然是
$$
P(D_c | \theta_{c}) = \prod_{x \in D_c} P(x | \theta_{c})
$$
对 $\theta_{c}$ 进行极大似然估计，就是寻找最大化似然 $P(D_c | \theta_{c})$ 的参数值 $\hat\theta_{c}$.

对数似然：解决（普通）似然的连乘操作容易造成下溢的问题。
$$
\begin{aligned}
LL(\theta_{c}) &= log P(D_c | \theta_{c}) \\
&= \sum_{x \in D_c}log P(x | \theta_{c}) \\
\end{aligned}
$$

此时参数 $\theta_{c}$ 的极大似然估计 $\hat\theta_{c}$ 是
$$
\hat\theta_{c} = arg max LL(\theta_{c})
$$

连续属性情况下，假设概率密度符合**正态分布**函数，则参数 $\mu_{c}$ 和 $\sigma_{c}^{2}$ 的极大似然估计是
$$
\begin{aligned}
\hat\mu_{c} &= \frac{1}{|D_c|}\sum_{x \in D_c}x \\
\hat\sigma_c^2 &= \frac{1}{|D_c|}\sum_{x \in D_c}(x - \hat\mu_c)(x - \hat\mu_c)^T
\end{aligned}
$$

### 朴素贝叶斯分类器
采用“属性条件独立性假设”：对已知类别，假设所有属性相互独立。

$$
P(c | x) = \frac{P(c) P(x | c)}{P(x)} = \frac{P(c)}{P(x)}\prod^d_{i=1}P(x_i | c)
$$

拉普拉斯平滑 Laplacian correction：避免其他属性携带的信息被训练集中未出现的属性值“抹去”。
$$
\begin{aligned}
\hat P(c) &= \frac{|D_c| + 1}{|D| + N} \\
\hat P(x_i | c) &= \frac{|D_{c, x_i}| + 1}{|D_c| + N_i}
\end{aligned}
$$

优点
1. 训练快，可以采用“懒惰学习”方式不进行任何预训练。
2. 在属性相互独立和多分类问题中效果好。
3. 可维护性高，即训练集的增删更新对性能影响小。

缺点
1. 属性相互独立前提难以满足。
2. 容易欠拟合

------

## K 近邻算法 KNN
给定测试样本，基于某种距离度量找出训练集中与其最靠近的 k 个训练样本，然后基于这 k 个邻居的信息来进行预测。通常，在分类任务中采用“投票法”，在回归任务中采用“平均法”。

### 步骤
1. 选定 K。
2. 计算测试点与其他所有样本点之间的“距离”。
3. 对距离排序并选出距离最小的 K 个邻居。
4. 收集最近邻的类别 Y。
5. 运用“投票法”或“平均法”得出分类/回归的结果。

### 说明

* 参数 K
K 越大分类边界越平滑。

* 距离 Distance
两个数据点相似性的度量，越相似则距离越大，且总是投影在 [0, 1] 之间。
常用距离函数：
1. 欧几里得距离 Euclidean Distance
在高纬表现不好。
$$
dist = \sqrt{\sum^p_{k=1}(a_k - b_k)^2}
$$
2. 曼哈顿距离 Manhattan Distance
$$
dist = \sum^n_{k = 1}|a_k - b_k|
$$
3. 名夫斯基距离 Minkowski Distance
$$
dist = (\sum^n_{k = 1}|x_k - y_k|^p)^{\frac{1}{p}}
$$
4. 余弦相似性 Cosine Similarity
$$
\begin{aligned}
cos(a, b) &= \frac{a \cdot b}{|a| \times |b|} \\
&= \frac{\sum^n_{k = 1}a_k \times b_k}{\sqrt{\sum^n_{k = 1}a_k^2} \times \sqrt{\sum^n_{k = 1}b_k^2}}
\end{aligned}
$$


* 正则化
1. Z-score Scaling
$$
X_{norm} = \frac{x - \mu}{\sigma}
$$
2. Min-Max Scaling
$$
X_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

* 优点
1. 无需训练。新数据对模型精确性无影响。
2. 容易实现，仅有 K 和 距离函数 两个参数。

* 缺点
1. 大数据集上表现不好，且计算距离消耗时间太多，性能下降。
2. 特征缩放（Feature Scaling）是必须的，不然可能得到错误结果。

------

## 逻辑回归（对数几率回归）logistics regression
### 概念

* 对数几率函数

$$
y = \frac{1}{1 + e^{-z}} \\
z = \vec{w}^T\vec{x} + b
$$

* 几率
转换得到 $\frac{y}{1 - y}$ 称为几率（odds），$ln(\frac{y}{1 - y})$ 即为对数几率（logit）。

$$
ln(\frac{y}{1 - y}) = \vec{w}^T\vec{x} + b
$$

* 损失函数

$$
l(\vec{w}, w_0|x,r) = -rlogy - (1 - r)log(1 - y)
$$

### 训练
Goal:
$$
min_w \ L(\vec{w})
$$

Iteration:
$$
\begin{aligned}
\vec{w}_{t+1} &= \vec{w}_t - \eta_t\frac{\partial L}{\partial w} \\
\frac{\partial{L}}{\partial{w_j}} &= -\sum_{l}(\frac{\partial L}{\partial y^{(l)}} \frac{\partial y^{(l)}}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial w_j}) \\
\end{aligned}
$$

梯度下降学习
$$
\begin{aligned}
y &= \frac{1}{1 + e^{-a}} \\
\frac{\partial y}{\partial a} &= y \times (1 - y) \\
\\
a &= w^Tx + w_0 \\
\frac{\partial a}{\partial w_j} &= x_j \\
\\
L(w, w_0 | D) &= -\sum^N_{l = 1}r^{(l)}logy^{(l)} + (1 - r^{(l)})log(1 - y^{(l)}) \\
\frac{\partial{L}}{\partial{w_j}} &= -\sum_l(\frac{r^{(l)}}{y^{(l)}} - \frac{1 - r^{(l)}}{1 - y^{(l)}})\frac{\partial y^{(l)}}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial w_j} \\
&= -\sum_l(\frac{r^{(l)}}{y^{(l)}} - \frac{1 - r^{(l)}}{1 - y^{(l)}}) y^{(l)}(1 - y^{(l)}) \frac{\partial a^{(l)}}{\partial w_j} \ \ (if \ y = sigmoid(a)) \\
&= -\sum_l(r^{(l)} - y^{(l)}) x_j^{(l)} = -\sum_l error \times input \\
\frac{\partial L}{\partial w_0} &= -\sum_l(r^{(l)} - y^{(l)})
\end{aligned}
$$

Softmax Function
$$
y_i = \hat{P}(C_i | x) = \frac{e^{w_i^Tx + w_{i0}}}{\sum^K_{j = 1}e^{w_j^Tx + w_{j0}}} \ \ i = 1, \dots, K
$$

------

## 支持向量机 Support Vector Machine
在高维或无限维空间中构造超平面或超平面集合，使得分类边界距离最近的数据点越远越好，从而缩小分类器的泛化误差。

### 线性 SVM
考虑以下形式点集：
$$
(\vec{x_1}, y_1), \dots, (\vec{x_n}, y_n)
$$
其中 $y_i$ 是 1 或者 -1，表明点 $\vec{x_i}$ 所属的类别。$\vec{x_i}$ 中每个都是一个 $p$ 维实向量。
需要寻找一个能将 $y_i = 1$ 的点集与 $y_i = -1$ 的点集分开的“最大间隔超平面”，使得超平面与最近的点 $\vec{x_i}$ 之间的距离最大化。
任何超平面可以写作：
$$
\vec{w} \cdot \vec{x} - b = 0
$$
其中 $\vec{w}$ 是法向量。参数 $\frac{b}{||\vec{w}||}$ 决定从原点沿法向量 $\vec{w}$ 到超平面的偏移量。

**硬间隔**
如果训练数据线性可分，选择分离两类数据的两个平行超平面，使得它们之间的距离尽可能大，在这两个超平面范围内的区域称为“间隔”，最大间隔超平面即使位于它们正中间的超平面，可以由方程：
$$
\vec{w} \cdot \vec{x} - b = 1
$$
或：
$$
\vec{w} \cdot \vec{x} - b = -1
$$
来表示。
即我们的优化目标可以写成：
$$
\begin{aligned}
&\underset{\vec{w}, b}{min} \quad \frac{1}{2}||\vec{w}||^2 \\
&s.t. \quad y_i(\vec{w}^T\vec{x_i} + b) \geq 1, \quad i = 1, 2, \dots, m.
\end{aligned}
$$

**软间隔**
将 SVM 拓展到数据线性不可分的情况，引入铰链损失函数，从而优化目标为：
$$
\underset{\vec{w}, b}{min} \quad \frac{1}{2}||\vec{w}||^2 + C\sum^m_{i = 1}L(y_i(\vec{w}^T\vec{x_i} + b) - 1)
$$

其中 L 是损失函数

* 0/1 损失函数
$$
l_{0/1}(z) = 
\begin{cases}
1, &if \ z \lt 0 \\
0, &otherwise \\
\end{cases}
$$

* hinge 损失函数
$$
L_{hinge}(z) = max(0, 1 - z)
$$

* 指数损失函数
$$
l_{exp}(z) = e^{-z}
$$

* 对率损失函数
$$
l_{log}(z) = log(1 + e^{-z})
$$

### 非线性分类
通过将**核技巧**应用于最大边界超平面来创建非线性分类器，即算法形式上类似但把**点积**换成了**非线性核函数**，将数据变换成高维空间的线性可分的。

核函数选择是支持向量机的最大变数，若核函数选择不合适，则意味着将样本映射到了一个不合适的特征空间，可能导致性能不佳。

常见核函数：

* 线性核
$$
\kappa(\vec{x_i}, \vec{x_j}) = \vec{x_i}^T\vec{x_j}
$$

* 多项式核
$$
\kappa(\vec{x_i}, \vec{x_j}) = (\vec{x_i}^T\vec{x_j})^d
$$

* 高斯核
$$
\kappa(\vec{x_i}, \vec{x_j}) = e^{(-\frac{||\vec{x_i} - \vec{x_j}||^2}{2\sigma^2})}
$$

* 拉普拉斯核
$$
\kappa(\vec{x_i}, \vec{x_j}) = e^{(-\frac{||\vec{x_i} - \vec{x_j}||^2}{\sigma})}
$$

* Sigmoid 核
$$
\kappa(\vec{x_i}, \vec{x_j}) = tanh(\beta\vec{x_i}^T\vec{x_j} + \theta)
$$

核函数性质：
1. 若 $\kappa_1$ 和 $\kappa_2$ 为核函数，则对于任意整数 $\gamma_1$、$\gamma_2$ ，其线性组合：$\gamma_1\kappa_1 + \gamma_2\kappa_2$ 也是核函数。
2. 若 $\kappa_1$ 和 $\kappa_2$ 为核函数，则对于核函数的直积：$\kappa_1\bigotimes\kappa_2(\vec{x}, \vec{z}) = \kappa_1(\vec{x}, \vec{z})\kappa_2(\vec{x}, \vec{z})$ 也是核函数。
3. 若 $\kappa_1$ 为核函数，则对于任意函数 $g(\vec{x})$：$\kappa(\vec{x}, \vec{z}) = g(\vec{x})\kappa_1(\vec{x}, \vec{z})g(\vec{z})$ 也是核函数。

### 计算与延伸
将其转换成对偶问题，其本身就是一个凸二次规划问题，能够直接用现成的优化计算包求解。还可以使用SMO、次梯度下降、坐标下降等现代方法求解。

SVM 潜在缺点：
* 需要对输入数据进行完全标记
* 未校准类成员概率
* SVM 仅直接适用于二分类。
* 模型参数很难理解，可解释性较差。

------

## 多层感知机 Multilayer Perceptron
一种前向结构的人工神经网路，映射一组输入向量到一组输出向量。MLP 可以看做一个有向图，由多个节点层组成，每一层都**全连接**到下一层。

### 激活函数
对神经元接收到的总输入值进行处理以产生神经元的输出，理想中的激活函数是 0-1 函数，即“1”对应于神经元兴奋，“0”对应于神经元抑制。但为了使用反向传播算法，激活函数必须为可微函数。

常见激活函数：
1. 阶跃函数
$$
sgn(x) = 
\begin{cases}
1, &\quad x \geq 0. \\
0, &\quad x \lt 0.
\end{cases}
$$
2. Sigmoid 函数
$$
sigmoid(x) = \frac{1}{1 + e^{(-x)}}
$$
3. 双曲正切函数
$$
tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}
$$
4. ReLU 函数
$$
ReLU(x) = max(0, x)
$$
5. Leaky ReLU 函数
$$
ReLU_{leaky}(x) = max(0.1x, x)
$$
6. Maxout 函数
$$
Maxout(x) = max(\vec{w_1}^Tx + b_1,\vec{w_2}^Tx + b_2)
$$
7. ELU 函数
$$
ELU(x) = 
\begin{cases}
x &\quad x \geq 0 \\
\alpha(e^x - 1) &\quad x \lt 0
\end{cases}
$$

### 训练
不断重复以下步骤直至收敛。

1. 正向传播
将输入向量应用到神经网络中，并更新所有层（隐藏层和输出层）的神经元的激活函数。
$$
\begin{aligned}
a_j &= \sum_i w_{ji}z_i \\
z_j &= \sigma(a_j) \\
\end{aligned}
$$
2. 反向传播
根据权重和损失函数导数来更新各神经元误差。
$$
\delta^L_j= \frac{\partial l}{\partial a_j^L}
$$
3. 更新参数
根据误差来计算各神经元的参数并更新。
$$
w_{ij}^{'} = w_{ij} + \eta\delta\sigma^{'}(a_j)z_i
$$

------
