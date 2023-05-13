import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
ex_ds=pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
impt=SimpleImputer()
Y=ex_ds.Price
X=ex_ds.drop(["Price"],axis=1)
X=X.select_dtypes(exclude=["object"])
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.6,test_size=0.4,random_state=0)
xr_train=pd.DataFrame(impt.fit_transform(x_train))
xr_test=pd.DataFrame(impt.fit_transform(x_test))
xr_train.columns=x_train.columns
xr_test.columns=x_test.columns
model=RandomForestRegressor(n_estimators=390,random_state=0)
model.fit(xr_train,y_train)
res=model.predict(xr_test)
print(res)
print(mean_absolute_error(res,y_test))
