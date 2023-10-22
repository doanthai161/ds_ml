import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data =  pd.read_csv(r'data_\data_.csv',header=None)
print(data)
#dữ liệu đang ở dạng dataframe nên chuyển thành mảng
X = data.values
'''
        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
          giá trị tung bình
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          ..tần số nhiều nhất, nếu 2 số cò tần suất xuất hiện bằng nhau thì lấy số đầu
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
'''
#missing_value : coi số còn trống là = 1, .., nan:' not a number '; kiểu strategy
imp = SimpleImputer(missing_values=np.nan, strategy='constant')
#cho dữ liệu vào
imp.fit(X)
#chuyển đổi dữ liệu
result =  imp.transform(X)
print(result)