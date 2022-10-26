# 导入机器学习使用的模块
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
# 导入可视化模块
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

# 1.导入数据库
data = pd.read_csv("./data/Social_Network_Ads.csv")
# 2.数据基本处理
# 2.1 数据分割
x = data[["Age","EstimatedSalary"]]
y = data["Purchased"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# test_size=0.25,即随机抽取0.25的原始数据作为测试集，剩余0.75的原始数据作为训练集

# 3.特征工程
#  3.1数据标准化
transfer1 = StandardScaler()
x_train = transfer1.fit_transform(x_train)
x_test = transfer1.fit_transform(x_test)

# 4机器学习（KNN）
estimator = KNeighborsClassifier(algorithm='kd_tree', n_jobs=2)
# 模型选择与调优——网格搜索和交叉验证
# 准备要调的超参数
param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=4)
estimator.fit(x_train,y_train)

# 5模型评估

y_pre = estimator.predict(x_test)
print("预测结果:\n", y_pre)
print("准确率为:\n", estimator.score(x_test, y_test))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pre))
# print("对比真实值和预测值:\n", y_test==y_pre)
print("在交叉验证中最好的结果为:\n", estimator.best_score_)
print("使用网格搜索的最好的模型:\n", estimator.best_estimator_)