from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
plt.show()

with open("data.txt", "w") as f:
    for i in range(X.shape[0]):
        f.write(f"{X[i][0]} {X[i][1]} {y[i]}\n")
