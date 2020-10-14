import numpy as np

pred_Y = np.genfromtxt('predicted_test_Y_nb.csv', delimiter = ',', dtype = np.float64)
Y = np.genfromtxt('train_Y_nb.csv', delimiter = ',', dtype = np.float64)
accuracy = 0
for i in range(len(Y)):
    if (Y[i] == pred_Y[i]):
        accuracy += 1
accuracy /= len(Y)
print(accuracy)