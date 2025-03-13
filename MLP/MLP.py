import numpy as np
from tensorflow.keras.datasets import mnist

class linear:

    def __init__(self, in_size, out_size):
        self.weights = np.random.randn(in_size, out_size)*0.01
        self.bias = np.random.randn(out_size)*0.01
        self.next = None
        self.pre = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights)+self.bias

    def backward(self, pdvalue, steps):
        self.pdvalue = np.dot(pdvalue, self.weights.T)
        self.pdweights = np.dot(self.x.T, pdvalue)
        self.pdbias = np.sum(pdvalue, axis=0)
        self.weights -= steps*self.pdweights+0.00005*self.weights
        self.bias -= steps*self.pdbias+0.00005*self.bias
        return self.pdvalue

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class activator:

    def __init__(self, tp ='relu'):
        self.type = tp
        self.next = None
        self.pre = None

    def func(self, x):
        if self.type == 'relu':
            return relu(x)
        if self.type == 'sigmoid':
            return sigmoid(x)
        if self.type == 'linear':
            return x

    def pd(self, x):
        if self.type == 'relu':
            return np.where(x > 0, 1, 0)
        if self.type == 'sigmoid':
            return sigmoid(x)*(1-sigmoid(x))
        if self.type == 'linear':
            return np.ones_like(x)

    def forward(self, x):
        self.x = x
        return self.func(x)

    def backward(self, pdvalue):
        self.pdvalue = pdvalue*self.pd(self.x)
        return self.pdvalue

class midLayer:

    def __init__(self, in_size, out_size, tp = 'relu'):
        self.linear = linear(in_size, out_size)
        self.activator = activator(tp)
        self.next = None
        self.pre = None

    def forward(self, x):
        self.x = x
        return self.activator.forward(self.linear.forward(x))

    def backward(self, pdvalue, steps):
        return self.linear.backward(self.activator.backward(pdvalue), steps)

def softmax(x):
    x = x-np.max(x, axis=1, keepdims=True)
    k = np.exp(x)
    return k/np.sum(k, axis=1, keepdims=True)

class outputLayer:

    def __init__(self, tp = 'softmax'):
        self.next = None
        self.pre = None
        self.type = tp

    def func(self, x):
        if self.type == 'softmax':
            return softmax(x)

    def forward(self, x):
        self.x = x
        return self.func(x)

    def backward(self, y_true, steps):
        y_pred = softmax(self.x)
        batch_size = y_true.shape[0]
        return (y_pred-y_true)/batch_size

def computeLoss(y_true, y_pred):
    log_probs = -np.log(np.clip(y_pred, 1e-10, 1.0))
    return np.sum(y_true * log_probs) / y_true.shape[0]

class MLP:

    def __init__(self, in_size, out_size, units, tp):
        if not(len(units) == len(tp)-2) or not tp:
            return ValueError
        units.append(out_size)
        units = [in_size]+units
        self.head = midLayer(units[0], units[1], tp[0])
        current = self.head
        for i in range(len(units)-2):
            current.next = midLayer(units[i+1], units[i+2], tp[i+1])
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

    def backward(self, x, steps):
        current = self.tail
        while current:
            x = current.backward(x, steps)
            current = current.pre
        return x

    def train(self, datax, datay, batch_size, num_samples, epoch, steps, losstype = 'computeLoss'):
        self.steps = steps
        datax = np.array(datax)
        datay = np.array(datay)
        for e in range(epoch):
            epochloss = 0
            permutation = np.random.permutation(num_samples)
            datax_shuffled = datax[permutation]
            datay_shuffled = datay[permutation]
            for i in range(0,len(datax),batch_size):
                x = datax_shuffled[i:i+batch_size]
                y = datay_shuffled[i:i+batch_size]
                y_pred = self.forward(x)
                if losstype == 'computeLoss':
                    loss = computeLoss(y, y_pred)
                    epochloss += loss*x.shape[0]
                self.backward(y, self.steps)
            print(f"Epoch {e+1}/{epoch} | Loss: {epochloss/num_samples:.4f}")

    def save(self, filename):
        weights = []
        biases = []
        current = self.head
        while current:
            if hasattr(current, 'linear'):
                weights.append(current.linear.weights)
                biases.append(current.linear.bias)
            current = current.next
        np.savez(filename, weights=np.array(weights, dtype=object), biases=np.array(biases, dtype=object))


    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
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

    mlp.train(x_train, y_train_onehot, batch_size, num_samples, epochs, learning_rate)

    mlp.save("mlp_model.npz")
    print("模型已保存。")

    mlp_loaded = MLP(784, 10, [128, 64], ['relu', 'relu', 'linear', 'softmax'])
    mlp_loaded.load("mlp_model.npz")
    print("模型已加载。")

    y_test_pred = mlp_loaded.forward(x_test)
    y_test_labels = np.argmax(y_test_pred, axis=1)
    accuracy = np.mean(y_test_labels == y_test)
    print(f"测试准确率：{accuracy * 100:.2f}%")