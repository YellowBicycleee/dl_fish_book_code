import numpy as np
import matplotlib.pylab as plt

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__' :
    x1 = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x1)
    plt.plot(x1, y1)
    plt.ylim(-0.1, 1.1)
    plt.show()