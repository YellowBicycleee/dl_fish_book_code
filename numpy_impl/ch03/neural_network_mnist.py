import sys, os
sys.path.append(os.pardir) # os.pardir: parent directory, '..'
import numpy as np
from dataset.mnist import load_mnist
# from PIL import Image
import pickle # pickle模块实现了基本的数据序列和反序列化

from sigmoid_function import sigmoid # my sigmoid
from softmax import softmax # my softmax

debug = True

def get_data() :
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # hidden layer 1
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    
    # hidden layer 2
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    # output layer
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

if __name__ == '__main__' :
    x, t = get_data()
    network = init_network()
    
    if debug:
        print(f'x.shape: {x.shape}')
        print(f'W1.shape: {network["W1"].shape}')
        print(f'W2.shape: {network["W2"].shape}')
        print(f'W3.shape: {network["W3"].shape}')
        
    
    accuracy_cnt = 0
    for i in range(len(x)) :
        y = predict(network, x[i])
        p = np.argmax(y) # get the index of the highest probability
        if p == t[i] :
            accuracy_cnt += 1
    
    print(f'accuracy_cnt = {accuracy_cnt}, len(x) = {len(x)}')
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
