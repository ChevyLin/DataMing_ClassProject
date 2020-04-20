import numpy as np
import matplotlib.pyplot as plt
import math

# input data
f = open("magic04.csv", "r")
row = f.readlines()
list = []
for i in range(len(row)):
    column_list = row[i].strip().split(",")
    column_list.pop()
    list.append(column_list)
a = np.array(list)
a = a.astype(float)  # convert to float

MeanVector = np.mean(a, axis=0)
centered = a-MeanVector  # centered

# compute dot product
dotProduct = np.dot(centered.T, centered)
print("The Dot Product is : ")
print(dotProduct/len(centered))

# compute out product
outProduct = 0
for i in range(len(centered)):
    outProduct = outProduct+centered[i].reshape(len(centered[0]), 1)*centered[i]
print("The Out Product is : ")
print(outProduct/len(centered))

# compute correlation through centered vector
t = centered.T
corr = np.corrcoef(t[0], t[1])
print("The Correlation is : ")
print(corr[0][1])

# Correlation Scatter Plot
picture = plt.figure()
ax1 = picture.add_subplot(111)  # 1*1 1pic
ax1.set_title("Correlation scatter plot")
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(t[0], t[1])
plt.show()


# pdf of normfunction
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


mu = np.mean(a, axis=0)[0]
sigma = np.var(a.T[0])
ax1 = picture.add_subplot(111)
ax1.set_title("Probability Density Function")

# Python实现正态分布
# 绘制正态分布概率密度函数
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
y_sig = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.vlines(mu, 0, np.exp(-(mu - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma), colors="c",
           linestyles="dashed")
plt.vlines(mu + sigma, 0, np.exp(-(mu + sigma - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma),
           colors="k", linestyles="dotted")
plt.vlines(mu - sigma, 0, np.exp(-(mu - sigma - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma),
           colors="k", linestyles="dotted")
plt.xticks([mu - sigma, mu, mu + sigma], ['μ-σ', 'μ', 'μ+σ'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Normal Distribution: $\mu = %.2f, $sigma=%.2f' % (mu, sigma))
plt.grid(True)
plt.show()

# compute variance
list = []
for i in range(len(a[0])):
    list.append(np.var(a.T[i]))
maxIndex = list.index(max(list))
minIndex = list.index(min(list))
print("The max variance is : ")
print(maxIndex+1)
print("The min variance is : ")
print(minIndex+1)


# compute Covariance
Cov = {}
for i in range(9):
    for j in range(i+1, 10):
        st = str(i+1)+'-'+str(j+1)
        Cov[st] = np.cov(a.T[i], a.T[j])[0][1]
print("The max covariance is : ")
print(max(Cov, key=Cov.get))
print("The min covariance is : ")
print(min(Cov, key=Cov.get))
