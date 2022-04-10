import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.style

data = pd.read_csv('USA_Housing.csv')


prepo = preprocessing.LabelEncoder()


Avg_Area_Income = prepo.fit_transform(data['Avg. Area House Age'])
Avg_Area_Number_of_Rooms = prepo.fit_transform(
    data['Avg. Area Number of Rooms'])
Avg_Area_Number_of_Bedrooms = prepo.fit_transform(
    data['Avg. Area Number of Bedrooms'])
Area_Population = prepo.fit_transform(data['Area Population'])
Price = prepo.fit_transform(data['Price'])
Address = prepo.fit_transform(data['Address'])


x = list(zip(Avg_Area_Income, Avg_Area_Number_of_Rooms,
         Avg_Area_Number_of_Bedrooms, Area_Population, Address))
y = list(Price)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(
#         f'\nPrediction: {predictions[x]}\nTest{x_test[x]}\nActual: {y_test[x]}')

p = 'Avg. Area House Age'
plt.scatter(data[p], data['Price'])
plt.xlabel(p)
plt.ylabel('Pricing')
plt.show()
