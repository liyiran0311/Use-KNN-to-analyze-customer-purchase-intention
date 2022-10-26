# �������ѧϰʹ�õ�ģ��
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
# ������ӻ�ģ��
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

# 1.�������ݿ�
data = pd.read_csv("./data/Social_Network_Ads.csv")
# 2.���ݻ�������
# 2.1 ���ݷָ�
x = data[["Age","EstimatedSalary"]]
y = data["Purchased"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# test_size=0.25,�������ȡ0.25��ԭʼ������Ϊ���Լ���ʣ��0.75��ԭʼ������Ϊѵ����

# 3.��������
#  3.1���ݱ�׼��
transfer1 = StandardScaler()
x_train = transfer1.fit_transform(x_train)
x_test = transfer1.fit_transform(x_test)

# 4����ѧϰ��KNN��
estimator = KNeighborsClassifier(algorithm='kd_tree', n_jobs=2)
# ģ��ѡ������š������������ͽ�����֤
# ׼��Ҫ���ĳ�����
param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=4)
estimator.fit(x_train,y_train)

# 5ģ������

y_pre = estimator.predict(x_test)
print("Ԥ����:\n", y_pre)
print("׼ȷ��Ϊ:\n", estimator.score(x_test, y_test))
print("��������:\n", confusion_matrix(y_test, y_pre))
# print("�Ա���ʵֵ��Ԥ��ֵ:\n", y_test==y_pre)
print("�ڽ�����֤����õĽ��Ϊ:\n", estimator.best_score_)
print("ʹ��������������õ�ģ��:\n", estimator.best_estimator_)