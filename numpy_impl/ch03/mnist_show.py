# coding: utf-8
import sys, os
sys.path.append(os.pardir) # os.pardir: parent directory, '..'
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))# 保存为numpy数组的图像数据转换为PIL用的图像数据
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)#flatten = true 读入的图像是一列numpy数组表示的

img = x_train[0]
label = t_train[0]
print(x_train.shape)
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

# img_show(img)
