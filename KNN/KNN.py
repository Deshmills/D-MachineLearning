import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import style

data = pd.read_csv('car.data')
print(data.head())

preprocess_car = preprocessing.LabelEncoder()
buying = preprocess_car.fit_transform(data['buying'])
maint = preprocess_car.fit_transform(data['maint'])
door = preprocess_car.fit_transform(data['door'])
persons = preprocess_car.fit_transform(data['persons'])
lug_boot = preprocess_car.fit_transform(data['lug_boot'])
safety = preprocess_car.fit_transform(data['safety'])
cls = preprocess_car.fit_transform(data['class'])

print(buying)
predict = 'class'


x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)


predicted = model.predict(x_test)
for x in range(len(predicted)):
    print('\nPredicted: ', predicted[x], '\nData: ',
          x_test[x], '\nActual: ', y_test[x])
    n = model.kneighbors([x_test[x]], 9, True)
    print('N: ', n)
