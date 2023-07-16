#python packages
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
# sample dataset load korlam just ekta
iris = sns.load_dataset('iris')
# print(iris)
# taking only those that i need
iris = iris[['petal_length', 'petal_width']]
# print(iris)

# training variable nebo
x = iris["petal_length"]
y = iris["petal_width"]
# print(x)

# showing the relation x and y
plt.scatter(x, y)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
# plt.show()

# training and test variables
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.4,
                                                    random_state = 23)

# here the "test size" determines how much of the total data I am using for my [n]
#[p] test purpose. Here 0.4 means I am using 40% of the total data
## random state = random state just takes data randomly. So, there is no bias.
# so it doesnot take the upper top rows or the bottom datas
# thus random state and test size helps us.
# print(X_test)

# print(X_train)

# machine learning model needs two dim data
# so we will take help from numpy to reshape the data
X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
# print(X_train)

#linear regreression initiate
lr = LinearRegression()

# fitting my train data into a regression frame
lr.fit(X_train, y_train)

# getting the training parameters m and c from
# y = mx + c
c = lr.intercept_
# print(c)
m = lr.coef_
# print(m)

# prediction
Y_Prediction = m * X_train + c
# print(Y_Prediction)

# prediction using builtin library
Y_predic = lr.predict(X_train)
# Y_predic

# Lets check if we got the best line here !
# showing the relation x and y
plt.plot(X_train, y_train, "+")
plt.plot(X_train, Y_predic, color = 'green')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# lets test the model if it can work in the same way using X test and Y test
# prediction
Y_Prediction_test = m * X_test + c
# print(Y_Prediction)


# showing the relation x and y
plt.plot(X_test, y_test, "+")
plt.plot(X_test, Y_Prediction_test, color = 'green')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
# plt.show()

 ### As the system has never seen the data of X test and Y test but still it works
