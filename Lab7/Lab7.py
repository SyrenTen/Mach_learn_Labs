import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

data_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
col_names = ["variance", "skewness", "curtosis", "entropy", "class"]
bankdata = pd.read_csv(data_link, names=col_names, sep=",", header=None)

sns.pairplot(bankdata, hue='class')
plt.show()

y = bankdata['class']
X = bankdata.drop('class', axis=1)

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)

svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, y_train)
y_pred_linear = svc_linear.predict(X_test)

svc_poly = SVC(kernel='poly', degree=3)
svc_poly.fit(X_train, y_train)
y_pred_poly = svc_poly.predict(X_test)

svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)
y_pred_rbf = svc_rbf.predict(X_test)

svc_sigmoid = SVC(kernel='sigmoid')
svc_sigmoid.fit(X_train, y_train)
y_pred_sigmoid = svc_sigmoid.predict(X_test)


def evaluate_model(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d').set_title(title)
    plt.show()
    print(classification_report(y_test, y_pred))


evaluate_model(y_test, y_pred_linear, 'Лінійна SVM')

evaluate_model(y_test, y_pred_poly, 'Поліноміальна SVM')

evaluate_model(y_test, y_pred_rbf, 'RBF SVM')

evaluate_model(y_test, y_pred_sigmoid, 'Сигмовидна SVM')

C_values = [0.1, 1, 10, 100]
for C in C_values:
    svc_rbf = SVC(kernel='rbf', C=C)
    svc_rbf.fit(X_train, y_train)
    y_pred = svc_rbf.predict(X_test)
    evaluate_model(y_test, y_pred, f'Матриця RBF SVM з C={C}')

gamma_values = [0.01, 0.1, 1, 10]
for gamma in gamma_values:
    svc_rbf = SVC(kernel='rbf', gamma=gamma)
    svc_rbf.fit(X_train, y_train)
    y_pred = svc_rbf.predict(X_test)
    evaluate_model(y_test, y_pred, f'Матриця RBF SVM з гамма={gamma}')
