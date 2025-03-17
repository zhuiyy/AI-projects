# -*-coding:utf-8-*-
'''
@ author: Zhuiy
'''

import numpy as np
from tensorflow.keras.datasets import mnist
import os


class linear:

    def __init__(self, in_size, out_size):
        self.weights = np.random.randn(in_size, out_size) * 0.01
        self.bias = np.random.randn(out_size) * 0.01
        self.next = None
        self.pre = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, pdvalue, steps, regular):
        self.pdvalue = np.dot(pdvalue, self.weights.T)
        self.pdweights = np.dot(self.x.T, pdvalue)
        self.pdbias = np.sum(pdvalue, axis=0)
        self.weights -= steps * self.pdweights + regular * self.weights
        self.bias -= steps * self.pdbias + regular * self.bias
        return self.pdvalue


def relu(x):
    return np.maximum(0, x)


def relupd(x):
    return np.where(x <= 0, 0, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidpd(x):
    return sigmoid(x) * (1 - sigmoid(x))


class activator:
    Atype = {'relu': relu, 'sigmoid': sigmoid, 'linear': (lambda x: x)}

    Apdtype = {'relu': relupd, 'sigmoid': sigmoidpd, 'linear': lambda x: 1}

    def __init__(self, tp='relu'):
        self.type = tp
        self.next = None
        self.pre = None
        self.func = self.Atype[self.type]
        self.pd = self.Apdtype[self.type]

    def forward(self, x):
        self.x = x
        self.sigmoidx = None
        k = self.func(x)
        if self.type == 'sigmoid':
            self.sigmoidx = k
        return k

    def backward(self, pdvalue):
        if self.type == 'sigmoid':
            self.pdvalue = pdvalue * self.sigmoidx * (1 - self.sigmoidx)
            return self.pdvalue
        self.pdvalue = pdvalue * self.pd(self.x)
        return self.pdvalue


class midLayer:

    def __init__(self, in_size, out_size, tp='relu'):
        self.linear = linear(in_size, out_size)
        self.activator = activator(tp)
        self.next = None
        self.pre = None

    def forward(self, x):
        self.x = x
        return self.activator.forward(self.linear.forward(x))

    def backward(self, pdvalue, steps, regualr):
        return self.linear.backward(self.activator.backward(pdvalue), steps, regualr)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    k = np.exp(x)
    return k / np.sum(k, axis=1, keepdims=True)


class outputLayer:
    Otype = {'softmax': softmax}

    # Opdtype = {'softmax' : softmaxpd} 因为本项目只是用softmax + computeloss输出，所以可以联合计算梯度，故不再细分类
    def __init__(self, tp='softmax'):
        self.next = None
        self.pre = None
        self.type = tp
        self.func = self.Otype[self.type]
        # self.pd = self.Opdtype[self.type]

    def forward(self, x):
        self.x = x
        return self.func(x)

    def backward(self, y_true, steps=None, regular=None):
        y_pred = softmax(self.x)
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size


def computeLoss(y_true, y_pred):
    log_probs = -np.log(np.clip(y_pred, 1e-10, 1.0))
    return np.sum(y_true * log_probs) / y_true.shape[0]


class MLP:

    def __init__(self, in_size, out_size, units, tp):
        if not (len(units) == len(tp) - 2) or not tp:
            return ValueError
        units.append(out_size)
        units = [in_size] + units
        self.head = midLayer(units[0], units[1], tp[0])
        current = self.head
        for i in range(len(units) - 2):
            current.next = midLayer(units[i + 1], units[i + 2], tp[i + 1])
            c = current
            current = current.next
            current.pre = c
        self.tail = outputLayer(tp[-1])
        current.next = self.tail
        self.tail.pre = current

    def forward(self, x):
        current = self.head
        while current:
            x = current.forward(x)
            current = current.next
        return x

    def backward(self, x, steps, regular):
        current = self.tail
        while current:
            x = current.backward(x, steps, regular)
            current = current.pre
        return x

    def train(
        self,
        datax,
        datay,
        batch_size,
        num_samples,
        epoch,
        steps,
        regular=0.0005,
        losstype='computeLoss',
    ):
        self.steps = steps
        self.regualr = regular
        datax = np.array(datax)
        datay = np.array(datay)
        for e in range(epoch):
            epochloss = 0
            permutation = np.random.permutation(num_samples)
            datax_shuffled = datax[permutation]
            datay_shuffled = datay[permutation]
            for i in range(0, len(datax), batch_size):
                x = datax_shuffled[i : i + batch_size]
                y = datay_shuffled[i : i + batch_size]
                y_pred = self.forward(x)
                if losstype == 'computeLoss':
                    loss = computeLoss(y, y_pred)
                    epochloss += loss * x.shape[0]
                self.backward(y, self.steps, self.regualr)
            print(f'Epoch {e+1}/{epoch} | Loss: {epochloss/num_samples:.4f}')

    def save(self, filename):
        module_dir = os.path.dirname(os.path.abspath(__file__))

        save_path = os.path.join(module_dir, filename)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        weights = []
        biases = []
        current = self.head
        while current:
            if hasattr(current, 'linear'):
                weights.append(current.linear.weights)
                biases.append(current.linear.bias)
            current = current.next

        np.savez(
            save_path,
            weights=np.array(weights, dtype=object),
            biases=np.array(biases, dtype=object),
        )

    def load(self, filename):
        module_dir = os.path.dirname(os.path.abspath(__file__))

        load_path = os.path.join(module_dir, filename)

        data = np.load(load_path, allow_pickle=True)
        weights = data['weights']
        biases = data['biases']

        current = self.head
        i = 0
        while current:
            if hasattr(current, 'linear'):
                current.linear.weights = weights[i]
                current.linear.bias = biases[i]
                i += 1
            current = current.next


if __name__ == '__main__':

    def one_hot_encode(y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0
    y_train_onehot = one_hot_encode(y_train, 10)
    y_test_onehot = one_hot_encode(y_test, 10)

    mlp = MLP(784, 10, [128, 64], ['relu', 'relu', 'linear', 'softmax'])

    batch_size = 64
    epochs = 10
    learning_rate = 0.1
    num_samples = x_train.shape[0]
    regular = 0.00005

    mlp.train(
        x_train, y_train_onehot, batch_size, num_samples, epochs, learning_rate, regular
    )

    mlp.save('mlp_model.npz')
    print('模型已保存。')

    mlp_loaded = MLP(784, 10, [128, 64], ['relu', 'relu', 'linear', 'softmax'])
    mlp_loaded.load('mlp_model.npz')
    print('模型已加载。')

    y_test_pred = mlp_loaded.forward(x_test)
    y_test_labels = np.argmax(y_test_pred, axis=1)
    accuracy = np.mean(y_test_labels == y_test)
    print(f'测试准确率：{accuracy * 100:.2f}%')
