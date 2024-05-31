import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('coffee_ratings.csv')

columns = ['total_cup_points', 'species', 'country_of_origin', 'variety', 'aroma', 'aftertaste',
           'acidity', 'body', 'balance', 'sweetness', 'altitude_mean_meters', 'moisture']
data = data[columns].dropna()

ord_enc = OrdinalEncoder()
for col in ['species', 'country_of_origin', 'variety']:
    data[col] = ord_enc.fit_transform(data[[col]])

X = data.drop('total_cup_points', axis=1)
y = data['total_cup_points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def train_evaluate(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, min_samples_split=2,
                       min_samples_leaf=1):
    clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae


n_estim_list = [10, 50, 100, 200, 500]
mse_results = []
mae_results = []
for n_est in n_estim_list:
    mse, mae = train_evaluate(X_train, X_test, y_train, y_test, n_estimators=n_est)
    mse_results.append(mse)
    mae_results.append(mae)
    print(f'MSE with {n_est} trees: {mse}')
    print(f'MAE with {n_est} trees: {mae}')

plt.plot(n_estim_list, mse_results, marker='o', label='MSE')
plt.plot(n_estim_list, mae_results, marker='o', label='MAE')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

max_depth_list = [None, 10, 20, 50, 100]
mse_results = []
mae_results = []
for max_depth in max_depth_list:
    mse, mae = train_evaluate(X_train, X_test, y_train, y_test, max_depth=max_depth)
    mse_results.append(mse)
    mae_results.append(mae)
    print(f'MSE max depth {max_depth}: {mse}')
    print(f'MAE max depth {max_depth}: {mae}')

plt.plot(max_depth_list, mse_results, marker='o', label='MSE')
plt.plot(max_depth_list, mae_results, marker='o', label='MAE')
plt.xlabel('Max Depth')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

min_sampl_split_list = [2, 5, 10, 20, 50]
mse_results = []
mae_results = []
for min_samples in min_sampl_split_list:
    mse, mae = train_evaluate(X_train, X_test, y_train, y_test, min_samples_split=min_samples)
    mse_results.append(mse)
    mae_results.append(mae)
    print(f'MSE min samples split {min_samples}: {mse}')
    print(f'MAE min samples split {min_samples}: {mae}')

plt.plot(min_sampl_split_list, mse_results, marker='o', label='MSE')
plt.plot(min_sampl_split_list, mae_results, marker='o', label='MAE')
plt.xlabel('Min Samples Split')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

min_samples_leaf_list = [1, 2, 5, 10, 20]
mse_results = []
mae_results = []
for min_leaf in min_samples_leaf_list:
    mse, mae = train_evaluate(X_train, X_test, y_train, y_test, min_samples_leaf=min_leaf)
    mse_results.append(mse)
    mae_results.append(mae)
    print(f'MSE min samples leaf {min_leaf}: {mse}')
    print(f'MAE min samples leaf {min_leaf}: {mae}')

plt.plot(min_samples_leaf_list, mse_results, marker='o', label='MSE')
plt.plot(min_samples_leaf_list, mae_results, marker='o', label='MAE')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()
