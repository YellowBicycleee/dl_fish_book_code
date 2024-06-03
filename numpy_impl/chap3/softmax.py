import numpy as np
import matplotlib.pylab as plt
# 
# def softmax(a) :
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 防止溢出
    y = exp_a / np.sum(exp_a)
    return y

if __name__ == '__main__' :
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)
    print(y)
    print(np.sum(y))