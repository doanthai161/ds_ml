# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# #đoạn code sinh ra dữ liệu
# # numOfPoint = 30
# # noise = np.random.normal(0,1,numOfPoint).reshape(-1,1)
# # x = np.linspace(30, 100, numOfPoint).reshape(-1,1)
# # N = x.shape[0]
# # y = 15*x + 8 
# # plt.scatter(x, y)

# data = pd.read_csv(r"data_\data_linear.csv").values
# N = data.shape[0]
# x = data[:, 0].reshape(-1, 1)
# y = data[:, 1].reshape(-1, 1)
# plt.scatter(x, y)
# plt.xlabel('met vuong')
# plt.ylabel('gia')
# #plt.show()

# #tạo mảng có kthuoc N, 1, dùng hstack nối với x-> tạo ma trận cột
# x = np.hstack((np.ones((N, 1), dtype=np.float64), x))
# print(x)
# #tạo mảng chứa 2 gtri 0.0, 1.0, reshape để biến thành ma trận với 1 cột
# w = np.array([0., 1.]).reshape(-1 ,1)
# print(w)
# #xác định số lần lặp
# numOfIteration = 100
# #tạo cost với 1 cột có 100 gtri 0, sdung để lưu dlieu sau mỗi lần lặp
# cost = np.zeros((numOfIteration, 1))
# learning_rate = 0.000001
# for i in range(numOfIteration):
#     r = np.dot(x, w) - y
#     cost[i] = 0.5*np.sum(r*r)
#     w[0] -= learning_rate*np.sum(r)
#     w[1] -= learning_rate*np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))
#     print(cost[i])

# predict = np.dot(x, w)
# plt.plot((x[0][1], x[N-1][1]),(predict[0], predict[N-1]), 'r')
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data_//data_square.csv').values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
print(x)
plt.scatter(x, y)
plt.xlabel('met vuong')
plt.ylabel('gia')
#plt.show()

lrg = LinearRegression()
lrg.fit(x,y)
y_pred = lrg.predict(x)
plt.plot((x[0], x[-1]),(y_pred[0], y_pred[-1]), 'r')
plt.show()

# Lưu nhiều tham số với numpy.savez(), định dạng '.npz'
np.savez('w2.npz', a=lrg.intercept_, b=lrg.coef_)
# Lấy lại các tham số trong file .npz
k = np.load('w2.npz')
lrg.intercept_ = k['a']
lrg.coef_ = k['b']
