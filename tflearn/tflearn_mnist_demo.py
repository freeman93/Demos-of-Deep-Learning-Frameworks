#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

# 获取数据，这里直接一个函数就把训练数据、测试数据全部分配好了，就是这么简单
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

# 定义神经网络
def build_model():
    # 重置所有参数和变量
    tf.reset_default_graph()

    # 定义输入层
    net = tflearn.input_data([None, 784])

    # 定义隐藏层
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 30, activation='ReLU')

    # 输出层
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model

# 构建模型
model = build_model()

# 训练模型
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=30)

# 测试模型
predictions = np.array(model.predict(testX)).argmax(axis=1)   # 预测值
actual = testY.argmax(axis=1)  # 真实值
test_accuracy = np.mean(predictions == actual, axis=0)   # 准确度
print("Test accuracy: ", test_accuracy)
