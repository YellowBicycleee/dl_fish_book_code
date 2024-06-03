import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)

if __name__ == '__main__' :
    #  = np.array([-1.0, 1.0, 2.0])
    x1 = np.arange(-5.0, 5.0, 0.1)
    y1 = relu(x1)
    plt.plot(x1, y1)
    # plt.ylim(-0.1, 1.1)
    # plt.xlim(-6.0, 6.0)
    plt.show()