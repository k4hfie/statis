import numpy as np
import matplotlib.pyplot as plt

study_time = np.array([1, 2, 3, 4, 5, 6, 7, 8])
score = np.array([52, 55, 61, 65, 68, 74, 78, 82])

def linear_regression(X, Y):
    sum_X = np.sum(X)
    sum_X2 = np.sum(X**2)
    sum_Y = np.sum(Y)
    sum_XY = np.sum(X*Y)
    n = np.size(X)

    
    b1 = (n * sum_XY - sum_X * sum_Y) / ( n * sum_X2 - sum_X**2)
    b0 = sum_Y/n - b1*(sum_X/n)

    def model(x):
        return b0 + b1*x

    return model

model = linear_regression(study_time, score)
predicted_score = model(study_time)

plt.figure(figsize=(12,9))
plt.plot(study_time, predicted_score, color='red')
plt.scatter(study_time, score)
plt.grid()
plt.xlabel('Study Time (h)')
plt.ylabel('Score')
plt.show()

print('STime\t Score \t Predicted Score')
for i in range(np.size(study_time)):
    print(f'{study_time[i]} \t {score[i]} \t {predicted_score[i]}')
    i = i + 1