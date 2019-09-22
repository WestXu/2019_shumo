# Environment

1. windows 7, 10
2. python 3.7
3. gurobi solver
4. `pip install -r requirements.txt`
5. `.\questions\2019年中国研究生数学建模竞赛F题`文件夹中放入两个原始数据文件`附件1：数据集1-终稿.xlsx`和`附件2：数据集2-终稿.xlsx`

# How to Run

1. 实例化求解器。which_dataset为数据集1或2。
```python
    from share_code import *
    s = Solver(which_dataset=1)
```

2. 自动添加约束
```python
    s.build_model()
```

3. 设置目标并开始求解。目标函数设置为`s.B.sum('*')`时，表示目标为最小路径，改为`s.Q.sum('*')`表示目标为最少校正次数。
```python
    # 创建目标函数
    s.model.setObjective(s.B.sum('*'), gurobipy.GRB.MINIMIZE)

    # 执行线性规划模型
    s.model.optimize()

    # 计算完成后beep提示
    winsound.Beep(500, 1000)
```

4. 保存结果，传入字符串为保存文件名。结果保存在项目目录的solutions文件夹，包括3D交互式图表(html格式)、结果变量矩阵(sol格式)、目标函数值(json格式)、其他衍生数据(csv格式)等等。
```python
    s.save('结果1')
```