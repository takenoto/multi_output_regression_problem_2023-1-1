# python -m free_code_camp_2020.support_vector_machine.svm
# SUPPORT VECTOR MACHINE
# SVM EXAMPLE
"""
- Effective high dimensional spaces
- Many kernel functions
- Classification and regression
"""


from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# split it  in features and labels
X = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
# test_size = 0.2 ==> Significa que 20% dos dados serão usados para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print('p: ', predictions)
print('a: ', y_test)
print('accuracy: ', acc)