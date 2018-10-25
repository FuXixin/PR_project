from sklearn.preprocessing import StandardScaler, Imputer
import numpy as np

# 缺值处理
def insert_nan(data):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data)
    return imp.transform(data)
    
# 标准化
def NL(data):

    scaler = StandardScaler()
    scaler.fit(data)
    scaler.transform(data)
    data = np.mat(data)
    return data
