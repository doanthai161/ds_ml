from sklearn.datasets import load_iris
#khởi tạo train *chia 75% trainning và 25% ktra
from sklearn.model_selection import train_test_split
#form sklearn.model_selection import train_test_split
import numpy as np
#khởi tạo cây quyết dịnh
from sklearn.tree import DecisionTreeClassifier
#lấy dữ liệu từ cây iris
iris_dataset = load_iris()
#trả về 4 tham số .... random_state: =0 khi chạy sẽ giữ nguyên kqua ramdom, = 1 chạy lại ra bộ trainning khác trong 150 iris
X_train,X_test,y_train,y_test = train_test_split(iris_dataset.data,iris_dataset.target, random_state=0)
#dặt cây quyết định
model = DecisionTreeClassifier()
#trainning
mymodel = model.fit(X_train,y_train)
X_new = np.array([[1.6, 2.3, 7.9, 9.2]])
#du doan
print(mymodel.predict(X_new))
