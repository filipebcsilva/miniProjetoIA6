import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


insurance_data = pd.read_csv("insurance.csv")

insurance_data["charges"] = np.log(insurance_data["charges"])

y = insurance_data["charges"]

lb = LabelEncoder()

categories_list = ["sex","smoker","region","children"]
insurence_encode = insurance_data.copy()

for category in categories_list:
    insurence_encode[category] = lb.fit_transform(insurance_data[category])

X = insurence_encode.drop("charges",axis = 1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=103)

linear_model = LinearRegression()
linear_model.fit(X_train,y_train)

y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_pred,y_test)
print("MSE:", mse)
score = linear_model.score(X_test,y_test)
print("SCORE:",score)
weight = linear_model.coef_
print("WEIGHT:",weight)
bias = linear_model.intercept_
print("BIAS:", bias)


fig, ax = plt.subplots(figsize=(20, 10))
plt.scatter(y_test,y_pred,color = "r")

coef = np.polyfit(y_test, y_pred, 1)
poly1d_fn = np.poly1d(coef)
x_vals = np.linspace(min(y_test), max(y_test), 100)
plt.plot(x_vals, poly1d_fn(x_vals), color="y")

plt.xlabel("charges_test")
plt.ylabel("charges_pred")
plt.show()
