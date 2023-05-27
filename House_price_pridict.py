#importing required libreries 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
#load data 
house_data=pd.read_csv("/content/house-price-data.zip")
#check the relation  of each colume with price
x_relation_check=house_data.copy()
impt=SimpleImputer()
orden=OrdinalEncoder()
x_relation_check=x_relation_check.dropna(axis=0)
y_relation_check=x_relation_check.pop("Price")
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
for col in x_relation_check.select_dtypes("object"):
  x_relation_check[col] , _ = x_relation_check[col].factorize()
discreat_=x_relation_check.dtypes==int
relation_ar=mutual_info_regression(x_relation_check,y_relation_check,discrete_features=discreat_)
relation_ar=pd.Series(data=relation_ar,name="miscore",index=x_relation_check.columns)
relation_ar=relation_ar.sort_values(ascending=False)
relation_ar=pd.DataFrame(relation_ar)
plt.figure(figsize=(17,8))
plt.title="relation graph"
sns.barplot(x=relation_ar["miscore"],y=relation_ar.index)
#from bargraph the address and area and no.of rooms are mostly related 



Y=house_data.Price
X=house_data.drop("Price",axis=1)
#exclude objects to train
X=X.select_dtypes(exclude=["object"])
#spliting data into two parts for testing and training
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.6,test_size=0.4,random_state=0)
xr_train=pd.DataFrame(impt.fit_transform(x_train))
xr_test=pd.DataFrame(impt.transform(x_test))
xr_train.columns=x_train.columns
xr_test.columns=x_test.columns
model=RandomForestRegressor(n_estimators=390,random_state=0)
model.fit(xr_train,y_train)
res=model.predict(xr_test)
print(res)
print(mean_absolute_error(res,y_test))
