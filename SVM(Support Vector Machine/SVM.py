import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

classes = ['malignant', 'benign']


clf2 = svm.SVC(kernel='poly', degree=2)
clf1 = KNeighborsClassifier(n_neighbors=9)
clf2.fit(x_train, y_train)


y_pred = clf2.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
