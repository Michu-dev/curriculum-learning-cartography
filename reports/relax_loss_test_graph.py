import matplotlib.pyplot as plt
import numpy as np

_lambda = 200
_epoch = 1

loss = np.ones((101))
difficulty = np.arange(0, 1.01, 0.01)


def relax_loss(loss, difficulty, epoch):
    return loss * (1 / _lambda ** (difficulty / epoch))


# x_diff = []
# for i in range(200):
#     x_diff.append(difficulty[np.random.randint(0, 10)])

x, y = [], []
for i in range(101):
    x.append(difficulty[i])
    y.append(relax_loss(loss[i], difficulty[i], _epoch))

plt.title(f"lambda = {_lambda}, epoch = {_epoch}")
plt.ylim(0.0, 1.2)
plt.xlabel("difficulty")
plt.ylabel("relaxed_loss")
plt.plot(x, y)

plt.show()
