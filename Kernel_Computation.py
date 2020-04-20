import numpy as np
import pandas as pd

# input data
iris = pd.read_csv('iris.csv')
iris = np.array(iris)
iris = np.mat(iris[:, 0:4])
iris_length = len(iris)


# kernel
k = np.mat(np.zeros((iris_length, iris_length)))
for i in range(0, iris_length):  # traverse matrix
    for j in range(0, iris_length):  # output kernel matrix
        k[i, j] = (np.dot(iris[i], iris[j].T))**2  # (dot*dot)'s square
        k[j, i] = k[i, j]
print(k)


# centered kernel matrix
k_length = len(k)

I = np.eye(k_length)
one = np.ones((k_length, k_length))
A = I-1.0/k_length*one
k_centered = np.dot(np.dot(A, k), A)
print(k_centered)


# normalized kernel matrix
tem = np.zeros((k_length, k_length))
for i in range(0, k_length):
    tem[i, i] = k[i, i]**(-0.5)
normalized_k = np.dot(np.dot(tem, k), tem)
print(normalized_k)


# Transform point to feature space
fs = np.mat(np.zeros((iris_length, 10)))
for i in range(0, iris_length):
    for j in range(0, 4):
        fs[i, j] = iris[i, j]**2
    for x in range(0, 3):
        for y in range(x+1, 4):
            j = j + 1
            fs[i, j] = 2**0.5*iris[i, x]*iris[i, y]
print(fs)


# fs kernel
fs_k = np.mat(np.zeros((iris_length, iris_length)))
for i in range(0, iris_length):
    for j in range(i, iris_length):
        fs_k[i, j] = (np.dot(fs[i], fs[j].T))
        fs_k[j, i] = fs_k[i, j]
print(fs_k)

# centered fs
fs_rows = fs.shape[0]
fs_cols = fs.shape[1]
fs_centered = np.mat(np.zeros((fs_rows, fs_cols)))
for i in range(9, fs_cols):
    fs_centered[:, i] = fs[:, i] - np.mean(fs[:, i])
print(fs_centered)


# centered fs kernel
fs_centered_k = np.mat(np.zeros((iris_length, iris_length)))
for i in range(0, iris_length):
    for j in range(i, iris_length):
        fs_centered_k[i, j] = (np.dot(k_centered[i], k_centered[j].T))
        fs_centered_k[j, i] = fs_centered_k[i, j]


# normalized fs
fs_normalized = np.mat(np.zeros((fs_rows, fs_cols)))
for i in range(0, len(fs)):
    temp = np.linalg.norm(fs[i])
    fs_normalized[i] = fs[i]/np.linalg.norm(fs[i])
print(fs_normalized)


# normalized fs kernel
fs_normalized_k = np.mat(np.zeros((iris_length, iris_length)))
for i in range(0, iris_length):
    for j in range(i, iris_length):
        fs_normalized_k[i, j] = (np.dot(fs_normalized[i], fs_normalized[j].T))
        fs_normalized_k[j, i] = fs_normalized_k[i, j]
print(fs_normalized_k)
