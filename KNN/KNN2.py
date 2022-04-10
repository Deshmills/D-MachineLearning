import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('USA_Housing.csv')

prepo = preprocessing.LabelEncoder()
Avg_Area_Income = prepo.fit_transform(data['Avg. Area Income'])
Avg_Area_House_Age = prepo.fit_transform(data['Avg. Area House Age'])
Avg_Area_Number_of_Rooms = prepo.fit_transform(
    data['Avg. Area Number of Rooms'])
Avg_Area_Number_of_Bedrooms = prepo.fit_transform(
    data['Avg. Area Number of Bedrooms'])
Area_Population = prepo.fit_transform(data['Area Population'])
Price = prepo.fit_transform(data['Price'])
Address = prepo.fit_transform(data['Address'])

x = list(zip(Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms,
         Avg_Area_Number_of_Bedrooms, Area_Population, Address))
y = list(Price)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

knn = KNeighborsClassifier(n_neighbors=2000)
knn.fit(x_train, y_train)
acc = knn.score(x_test, y_test)
print(acc)
