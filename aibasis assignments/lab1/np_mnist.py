# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


import numpy as np
from tqdm  import tqdm


X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


def relu(x): # 64 * 256 -> 64 * 256
    return np.where(x >= 0, x, 0)

def relu_prime(previousx, relux=True):
    return np.where(previousx >=0, 1, 0)

def sigmoid(x): # 64 * 256 -> 64 * 256
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x, sigmoidx):
    if not sigmoidx:
        x = sigmoid(x)
    return x * (1 - x)

def tanh(x):
    return (1 - np.exp(-2 * x))/(1 + np.exp(-2 * x))

def sigmoid_prime(x, tanhx):
    if not tanhx:
        x = tanh(x)
    return 1 - x * x


def softmax(x): # 64 * 10 -> 64 * 10
    x = x - np.max(x, axis=1, keepdims=1)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=1)

def softmax_prime(x, batch_size, softmaxx): # 64 * 10 ->  64 * 10 * 10
    if not softmaxx:
        x = softmax(x)
    x = np.reshape(x, (batch_size, 10, 1)) # 64 * 10 * 1
    I = np.eye(10).reshape(1, 10, 10) # 1 * 10 * 10
    return x * I - x @ x.transpose(0,2,1) # 64 * 10 * 10
    

def compute_loss(y_true, y_pred): # 64 * 10 -> 64 * 1
    y_pred = np.clip(y_pred, 1e-15, 1) 
    return -np.sum(y_true * np.log(y_pred), axis=1)

def compute_loss_prime(y_true, y_pred): # 64 * 10 -> 64 * 10
    y_pred = np.clip(y_pred, 1e-15, 1) 
    return -y_true / y_pred

def square_loss(y_true, y_pred): # 64 * 10 -> 64 * 1
    return np.sum(np.square(y_true - y_pred), axis=1)

def square_loss_prime(y_true, y_pred): # 64 * 10 -> 64 * 10
    return 2 * y_pred - 2 * y_true


def init_weights(shape=()):
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)



class Network_2(object):

    Atype = {'relu':relu, 'sigmoid':sigmoid}
    A_ptype = {'relu':relu_prime, 'sigmoid':sigmoid_prime}
    Ltype = {'compute':compute_loss, 'square':square_loss}
    L_ptype = {'compute':compute_loss_prime, 'square':square_loss_prime}

    def __init__(self, input_size, hidden_size, output_size, lr, atype, ltype, retype, rer):
        self.W1 = init_weights(shape=(input_size, hidden_size))
        self.W2 = init_weights(shape=(hidden_size, output_size))
        self.b1 = init_weights(shape=(hidden_size, 1))
        self.b2 = init_weights(shape=(output_size, 1))
        self.Activator = self.Atype[atype]
        self.atype = atype
        self.Activator_prime = self.A_ptype[atype]
        self.Loss = self.Ltype[ltype]
        self.ltype = ltype
        self.Loss_prime = self.L_ptype[ltype]
        self.lr = lr
        self.re = retype
        self.rer = rer
    def forward(self, x):
        self.X1 = x
        self.lX1 = np.dot(self.X1, self.W1) + self.b1.T
        self.X2 = self.Activator(self.lX1)
        self.lX2 = np.dot(self.X2, self.W2) + self.b2.T
        self.y_pred = softmax(self.lX2)
        return self.y_pred
    
    def backward(self, y_pred, y_true, batch_size):
        if self.ltype != 'compute':
            dp = self.Loss_prime(y_true, y_pred) # 64 * 10
            dlX2 = np.einsum('ij,ijk->ik', dp, softmax_prime(self.y_pred, batch_size, True)) # 64 * 10
        else:
            dlX2 = y_pred - y_true
        db2 = np.sum(dlX2, axis=0, keepdims=1).T / batch_size # 10
        dW2 = np.dot(self.X2.T, dlX2) / batch_size # 256 * 10 (求和工作由点积进行)
        dX2 = np.dot(dlX2, self.W2.T) # 64 * 256

        if self.atype == 'relu':
            dlX1 = self.Activator_prime(self.lX1, True) * dX2
        else:
            dlX1 = self.Activator_prime(self.X2, True) * dX2 # 64 * 256 
        db1 = np.sum(dlX1, axis=0, keepdims=1).T / batch_size # 256
        dW1 = np.dot(self.X1.T, dlX1) / batch_size # 784 * 256

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        if self.re == 'L2':
            self.W2 -= self.rer * self.W2 * 2 * self.lr
            self.b2 -= self.rer * self.b2 * 2 * self.lr 
            self.W1 -= self.rer * self.W1 * 2 * self.lr
            self.b1 -= self.rer * self.b1 * 2 * self.lr
        elif self.re == 'L1':
            self.W2 -= self.rer * np.where(self.W2 >= 0, 1, -1)
            self.b2 -= self.rer * np.where(self.b2 >= 0, 1, -1)
            self.W1 -= self.rer * np.where(self.W1 >= 0, 1, -1)
            self.b1 -= self.rer * np.where(self.b1 >= 0, 1, -1)

    def step(self, x_batch, y_batch, batch_size):
        y_pred = self.forward(x_batch)
        loss = np.sum(self.Loss(y_batch, y_pred), axis=0) / batch_size
        self.backward(y_pred, y_batch, batch_size)
        return loss

def check(a):
    print(np.shape(a))

if __name__ == '__main__':
    try:
        a = int(input('use the best setting?(1/0)'))
        if a == 0:
            hidden_size = int(input('hidden_size:'))
            batch = int(input('batch_size:'))
            E = int(input('epoch_times:'))
            lr = float(input('learning_rate:'))
            lrd = float(input('lerning_rate_descent'))
            i = input('please choose your activator among:\n    relu:0\n    sigmoid:1\n    tanh:2\n')
            atype = {'0':'relu', '1':'sigmoid', '2':'tanh'}[i]
            i = input('please choose your loss function between:\n    square:0\n    compute:1\n')
            ltype = {'0':'square', '1':'compute'}[i]
            i = input('please choose your regulation between:\n    None:0\n    L1:1\n    L2:2\n')
            retype = {'0':None, '1':'L1', '2':'L2'}[i]
            if i == 0:
                rer = 0
            else:
                rer = float(input('regulation_rate:'))
        elif a == 1:
            hidden_size= 300
            batch = 64
            lr = 0.13
            lrd = 1.1
            E = 10
            atype = 'relu'
            ltype = 'compute'
            retype = 'L2'
            rer = 0.00005
        else:
            raise ValueError('input error')
    except Exception:
        raise ValueError('input error')
    net = Network_2(784, hidden_size, 10, lr, atype, ltype, retype, rer)
    preal = -1
    for epoch in range(E):
        # losses = []
        # accuracies = []
        shuffle = np.random.permutation(len(X_train))
        X_train = X_train[shuffle]
        y_train = y_train[shuffle]
        p_bar = tqdm(range(0, len(X_train), batch))
        loss_epoch = 0
        count = 0
        for i in p_bar:
            count += 1
            X = X_train[i:i+64:] # 64 * 784
            y = y_train[i:i+64:] # 64 * 10
            actual_batch_size = X.shape[0]
            loss_epoch += net.step(X, y, actual_batch_size)
        accuracy_epoch = np.mean(np.argmax(net.forward(X_val), axis=1) == np.argmax(y_val, axis=1))
        loss_epoch /= count
        print(f'epoch {epoch + 1} summary : loss = {loss_epoch:.3f} | val_accuracy = {accuracy_epoch * 100:.3f}% | learning rate = {net.lr:.3f}')
        if preal < loss_epoch:
            net.lr /= lrd
        preal = loss_epoch
    print()
    print('trained')
    print(f'test_accuracy:{np.mean(np.argmax(net.forward(X_test), axis=1) == np.argmax(y_test, axis=1)) * 100:.3f}%')
        