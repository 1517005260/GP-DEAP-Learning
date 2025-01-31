# 基于DEAP的遗传编程（Genetic Programming，GP）系列教程

本系列教程主要介绍如何基于DEAP实现一些流行的遗传编程概念，包括：

* 单树GP
* 多树GP
* 多目标GP
* 集成学习
* 启发式算法生成

遗传编程文章：https://blog.csdn.net/qq_46450354/article/details/128419169

环境配置：

```bash
(base) glk@ggg:~/project/DEAP-GP-Tutorial$ pip show deap
Name: deap
Version: 1.4.2
Summary: Distributed Evolutionary Algorithms in Python
Home-page: https://www.github.com/deap
Author: deap Development Team
Author-email: deap-users@googlegroups.com
License: LGPL
Location: /home/glk/project/anaconda3/lib/python3.11/site-packages
Requires: numpy
Required-by: 
```

上述概念通过以下示例实现：

1. [基于单树GP的符号回归（Symbolic Regression）](application/symbolic-regression.ipynb)
2. [基于多树GP的特征工程（Feature Construction）](application/feature-construction.ipynb)
3. [基于多目标GP的符号回归 （Multi-Objective Symbolic Regression）](application/multiobjective-sr.ipynb)
4. [基于GP的集成学习（Ensemble Learning）](application/ensemble-learning.ipynb)
5. [基于GP的旅行商问题规则生成（TSP）](application/TSP.ipynb)
6. [为什么使用GP而不是神经网络？（Feature Construction）](application/cross-validation-score.ipynb)
6. [基于GP自动设计优化算法-实验篇（自动生成北极狐算法）](application/automatically-design-de-operators.ipynb)
7. [基于GP自动设计优化算法-理论证明篇（自动生成北极狐算法）](application/theoretical_analysis.ipynb)
8. [基于不同算子集的多树GP](application/multisets_gp.ipynb)

同时，本教程包含了一些工程技巧：

1. [基于Numpy实现向量化加速](tricks/numpy-speedup.ipynb)
2. [基于PyTorch实现GPU加速](tricks/pytorch-speedup.ipynb)
3. [基于手动编写编译器实现加速](tricks/compiler-speedup.ipynb)
4. [基于Numba实现Lexicase Selection加速](tricks/numba-lexicase-selection.ipynb)
5. [基于多进程实现异步并行评估](tricks/multiprocess_speedup.md)

此外，DEAP还有一些注意事项：

1. [VarAnd和VarOr算子的使用注意事项](operator/varor-varand.ipynb)
2. [Crossover算子的注意事项](operator/crossover.ipynb)
2. [Lexicase Selection算子的注意事项](operator/lexicase-selection.ipynb)


### 遗传编程与传统遗传算法的对比

遗传编程（Genetic Programming, GP）是一种基于进化计算的算法，它的目标是自动生成计算机程序以解决特定问题。遗传编程通过模拟自然选择过程，利用树形结构表示程序，逐步演化出性能更优的程序。每个个体代表一个可能的程序结构，节点表示操作符（如加、减、乘、除等），叶子节点表示操作数或变量。通过选择、交叉、变异等操作，遗传编程不断优化程序结构。

遗传算法（Genetic Algorithm, GA）则是一种优化算法，主要用于寻找问题的最佳解决方案。遗传算法中的个体通常表示为固定长度的字符串（如二进制字符串或实数向量），代表问题的潜在解，主要应用于参数优化等领域。

### 遗传编程与传统遗传算法的关系与区别

| 特征            | 遗传编程（GP）                                  | 传统遗传算法（GA）                            |
|-----------------|--------------------------------------------------|-----------------------------------------------|
| 表示方式     | 程序结构，通常为树形结构，节点为操作符，叶子节点为操作数或变量 | 固定长度的字符串（如二进制字符串或实数向量）  |
| 主要目标   | 自动生成和优化程序，处理复杂问题，如自动生成算法、符号回归等 | 主要用于优化问题，寻找最优参数组合           |
| 应用领域     | 自动生成程序、符号回归、机器学习模型构建等       | 函数优化、调度问题等需要找到最优参数的场景    |
| 操作方式    | 选择、交叉（交换子树）、变异等操作               | 选择、交叉、变异等操作，但操作的对象是字符串  |
| 复杂度      | 能处理更复杂的问题，如程序结构的自动生成和优化   | 适用于较为简单的优化问题，个体为固定结构      |

遗传编程是遗传算法的扩展，专注于程序结构的自动生成和优化，能够处理更复杂的任务。遗传算法则更多地应用于参数优化等领域，个体的表示方式较为简单，通常为字符串形式。