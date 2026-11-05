# check for residual normality
# check for homoscedasticity

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

data = np.loadtxt('datawow.csv', delimiter=',', skiprows=1)

tempurature = data[:, 2]
polution = data[:, 1]

def linear_regression(X, Y):
    n = np.size(X)
    sum_X = np.sum(X)
    sum_X2 = np.sum(X**2)
    sum_Y = np.sum(Y)
    sum_XY = np.sum(X*Y)
    
    b1 = (n * sum_XY - sum_X * sum_Y) / ( n * sum_X2 - sum_X**2)
    b0 = sum_Y/n - b1*(sum_X/n)

    def model(x):
        return b0 + b1*x

    return model, b1

def r_squared(Y, predicted_Y):
    n = np.size(Y)
    mean = np.sum(Y) / n
    residual_sum, total_sum = 0, 0

    # sum of squares residuals
    residual_sum = np.sum((Y - predicted_Y)**2)

    # total sum of squares
    total_sum = np.sum((Y - mean)**2)

    error = 1 - (residual_sum/total_sum) 
    return error

def rmse(Y, predicted_Y):
    n = np.size(Y)
    sum = np.sum((Y-predicted_Y)**2)

    error = np.sqrt(sum/n)
    return error

def print_data(X, Y, predicted_Y):
    print('X\t Y \t Predicted Y')

    for i in range(np.size(X)):
        print(f'{X[i]} \t {Y[i]} \t {predicted_Y[i]}')

def test_polution(b1, X, Y, Predicted_Y):
    n = np.size(Y)
    mean_X = np.mean(X)

    residual_variance = np.sum((Y-Predicted_Y)**2) / (n-2)
    squared_deviation = np.sum((X-mean_X)**2)
    standard_error = np.sqrt(residual_variance/squared_deviation)

    polution = b1 / standard_error
    return polution


model, b1 = linear_regression(tempurature, polution)
predicted_polution = model(tempurature)
r2 = r_squared(polution, predicted_polution)
rmse = rmse(polution, predicted_polution)
t_polution = test_polution(b1, tempurature, polution, predicted_polution)

p_value = 2*(t.sf(np.abs(t_polution), df=np.size(polution)-2))
print(p_value)

plt.figure(figsize=(12,9))
plt.plot(tempurature, predicted_polution, color='red', label ='Prediction Model')
plt.scatter(tempurature, polution, color='blue', label = 'Measured Data')
plt.legend(fontsize=12)
plt.xlabel('Tempurature (°C)', fontsize=12)
plt.ylabel('Polution (AQI)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.title(f'R^2 = {format(r2, '.4g')}, RMSE = {format(rmse, '.4g')}',fontsize=15, fontweight='bold')
plt.show()