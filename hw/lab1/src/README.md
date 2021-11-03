# lab1
#### id: 519021910861
#### name: huidong xu
------

## 运行说明
我是在 pycharm 中的虚拟环境中运行的，直接用 pycharm 打开 src 文件夹，在文件 $bayes.py$ 中点击主函数旁边的运行按钮即可运行用 $sklearn$ 实现的高斯朴素贝叶斯方法，在文件 $logi.py$ 中点击主函数旁边的运行按钮即可运行自己实现的逻辑回归方法。

输出均打印在控制台。

特别说明：如果想看逻辑回归中每 100 步迭代的准确率，请更改 **主函数** 中这一行：

```python
  d = logistic_model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.12, print_cost=True, show_iteration_accuracy=False)
```

将其 $learning\_rate$ 改变为 $0.01$，将 $show_iteration_accuracy$ 改变为 $True$，更改后为：

```python
  d = logistic_model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.01, print_cost=True, show_iteration_accuracy=True)
```

## pycharm 安装包小技巧
更换源
``` 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name 
```
豆瓣
http://pypi.douban.com/simple/
阿里   
http://mirrors.aliyun.com/pypi/simple/   
中国科学技术大学
http://pypi.mirrors.ustc.edu.cn/simple/   