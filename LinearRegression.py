import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


data = pd.read_csv('./Cellphone.csv')


new_cols = ['Product_id','Sale','weight','resoloution','ppi','cpu core','cpu freq','internal mem','ram','RearCam','Front_Cam','battery','thickness','Price']

data = data[new_cols]

data.drop(['Product_id'],axis=1,inplace=True)




dt_Train , dt_Test = train_test_split(data , test_size=0.3 , shuffle=False)

k = 4; 
kf = KFold(n_splits=k , random_state=None)


def error(y_pred , y):
	difArray = []
	y_array = np.array(y)
	for i in range(0 , len(y_pred)):
		dif = np.abs(y_array[i] - y_pred[i])
		difArray.append(dif)

	return np.mean(difArray)     


min = 999999999999999999999999

i = 1;


for(train_index , validation_index) in kf.split(dt_Train):

	X_train = dt_Train.iloc[train_index,:12]
	y_train = dt_Train.iloc[train_index,12]

	X_val = dt_Train.iloc[validation_index,:12]
	y_val = dt_Train.iloc[validation_index,12]


	lr = LinearRegression()

	lr.fit(X_train, y_train)

	y_pred_train = lr.predict(X_train)
	y_pred_val = lr.predict(X_val)

	sum_error = error(y_pred_train, y_train) + error(y_pred_val, y_val)


	if( sum_error < min):
		min = sum_error
		regr = lr.fit(X_train, y_train)
		last = i
	i = i + 1; 


print("w = " , regr.coef_)
print("\nw0 = " , regr.intercept_)
print("\nKết quả tối ưu thu được nằm ở lần thử thứ " , last)

y_test = dt_Test.iloc[:,12]
X_test = dt_Test.iloc[:,:12]
y_pred_test = regr.predict(X_test)

print("\nCoefficient %.3f"   %regr.score(X_test , y_test))
