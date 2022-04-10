import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

print(data.head())

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

'''
best = 0
for _ in range(25):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        with open('studentgrades.pickle', 'wb') as f:
            pickle.dump(linear, f)'''

pickle_in = open('studentgrades.pickle', 'rb')

linear = pickle.load(pickle_in)


print(f'Coefficient: {linear.coef_}')
print(f'Intercept: {linear.intercept_}')

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use('ggplot')
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()
