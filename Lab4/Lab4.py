# Варіант 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

x = np.arange(0, 10, 0.1)
y = x**3 - 10 * x**2 + x

x_train = x[::2].reshape(-1, 1)
y_train = y[::2]

polynom = PolynomialFeatures(degree=13)

model = make_pipeline(polynom, LinearRegression())
model.fit(x_train, y_train)
y_pred = model.predict(x.reshape(-1, 1))

plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Дата', color='red')
plt.plot(x, y, label='Функція', color='green')
plt.plot(x, y_pred, label='Поліном регресія (13 ступень)', color='blue')
plt.legend()
plt.title('Поліном регресія без регулярізації ')
plt.show()

mse = mean_squared_error(y, y_pred)
print(f'Середня помилка (без регулярізації): {mse}')

model_ridge = make_pipeline(polynom, Ridge(alpha=1.0))
model_ridge.fit(x_train, y_train)
y_pred_ridge = model_ridge.predict(x.reshape(-1, 1))

plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Дата', color='red')
plt.plot(x, y, label='Функція', color='green')
plt.plot(x, y_pred_ridge, label='Ridge регрес', color='blue')
plt.legend()
plt.title('Регресія з регулярізацією L2')
plt.show()

mse_ridge = mean_squared_error(y, y_pred_ridge)
print(f'Квадратична помилка з регулярізацією L2: {mse_ridge}')

model_lasso = make_pipeline(polynom, Lasso(alpha=1e-3, max_iter=10000))
model_lasso.fit(x_train, y_train)
y_pred_lasso = model_lasso.predict(x.reshape(-1, 1))

plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Дата', color='red')
plt.plot(x, y, label='Функція', color='green')
plt.plot(x, y_pred_lasso, label='Lasso регресія', color='blue')
plt.legend()
plt.title('Регресія з резулярізацією L1 ')
plt.show()

mse_lasso = mean_squared_error(y, y_pred_lasso)
print(f'Помилка з резулярізації L1: {mse_lasso}\n')

print(f'Середня помилка (без регулярізації): {mse}')
print(f'Квадратична помилка з регулярізацією L2: {mse_ridge}')
print(f'Помилка з резулярізацією L1: {mse_lasso}')
# Виходить так, що модель без регуляризації точно відтворює функцію, але ймовірно перетренована.
# Модель з L2 краще через точність й вона наче уникає перетренування.
# У моделі з L1 найвище значення, але вона не покращує результат так як усі характеристики є важливими.
