import numpy as np
import matplotlib.pylab as plt


def step_function(x) :
    return np.array(x > 0, dtype=np.integer)        # donnot use np.int, it will cause error

if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    print(step_function(x))
    
    x1 = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x1)
    plt.plot(x1, y1)
    plt.ylim(-0.1, 1.1)
    plt.show()