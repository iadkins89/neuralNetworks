import numpy as np

class three_layer_NN:
    def __init__(self, X, y, layer1=1, layer2=1, layer3=1):
        self.X = X
        self.y = y
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

        self.inputs = X.shape[1]
        self.w1 = np.random.rand(self.inputs, self.layer1)
        self.b1 = np.random.rand(self.layer1)

        self.w2 = np.random.rand(self.layer1, self.layer2)
        self.b2 = np.random.rand(self.layer2)

        self.w3 = np.random.rand(self.layer2, self.layer3)
        self.b3 = np.random.rand(self.layer3)

    def __sigmoid(self, x, derivative=False):
        if not derivative: return 1 / (1 + np.exp(-1 * x))
        else: return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def __relu(self, x, derivative=False):
        if not derivative:
            if x.any() > 0: return x
            else: return 0
        else:
            if x.any() > 0: return 1
            else: return 0

    def __indentity(self, x): return x

    def __loss(self, o, y): return np.square(o - y)

    def __get_hiddenserializer(self):
        if self.hidden_activation == 'relu':
            return self.__relu
        elif self.hidden_activation == 'sigmoid':
            return self.__sigmoid
        else:
            raise ValueError(self.hidden_activation)

    def __hiddenserialize(self, x, derivative=False):
        serialize = self.__get_hiddenserializer()
        return serialize(x, derivative=derivative)

    def __get_outputserializer(self):
        if self.output_activation == 'relu':
            return self.__relu
        elif self.output_activation == 'sigmoid':
            return self.__sigmoid
        elif self.output_activation == 'none':
            return self.__indentity
        else:
            raise ValueError(self.output_activation)

    def __outputserialize(self, x):
        serialize = self.__get_outputserializer()
        return serialize(x)

    def __backprop(self, x, y, z1, a1, z2, a2, a3):
        d3 = (a3 - y)
        d2 = d3.dot(self.w3.T)*self.__hiddenserialize(z2, derivative=True)
        d1 = d2.dot(self.w2.T)*self.__hiddenserialize(z1, derivative=True)

        self.w3 = self.w3 - (self.learn_rate*np.outer(d3,a2)).T
        self.b3 = self.b3 - self.learn_rate*d3

        self.w2 = self.w2 - (self.learn_rate*np.outer(d2,a1)).T
        self.b2 = self.b2 - self.learn_rate*d2

        self.w1 = self.w1 - (self.learn_rate*np.outer(d1, x)).T
        self.b1 = self.b1 - self.learn_rate*d1

    def __forward(self, x):
        z1 = x.dot(self.w1) + self.b1
        a1 = self.__hiddenserialize(z1)

        z2 = a1.dot(self.w2) + self.b2
        a2 = self.__hiddenserialize(z2)

        a3 = a2.dot(self.w3) + self.b3
        output = self.__outputserialize(a3)

        return output
    def __forward_back(self, x, y):
        z1 = x.dot(self.w1) + self.b1
        a1 = self.__hiddenserialize(z1)

        z2 = a1.dot(self.w2) + self.b2
        a2 = self.__hiddenserialize(z2)

        a3 = a2.dot(self.w3) + self.b3
        output = self.__outputserialize(a3)

        self.__backprop( x, y, z1, a1, z2, a2, output)

        return output

    def train(self, num_epochs, learn_rate, hidden_activation, output_activation, verbose=True):
        self.learn_rate = learn_rate
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        cost = []
        predictions =[]
        for epoch in range(0, num_epochs):
            for i in range(len(self.X)):
                o = self.__forward_back(self.X[i], self.y[i])
                cost.append(self.__loss(o, self.y[i]))

                if epoch == range(0, num_epochs)[-1]:
                    predictions.append(o)
            if verbose:
                print(f'Epoch {epoch}| Acc {(1 - (sum(cost) / len(cost))) * 100} | Error {(sum(cost) / len(cost))*100}')
        if not verbose:
                print(f'Acc {(1 - (sum(cost) / len(cost))) * 100} | Error {(sum(cost) / len(cost)) * 100}')

        return predictions, cost

    def predict(self, xt):
        predictions = []
        for i in range(len(xt)):
            predictions.append(self.__forward(xt[i]))

        return predictions
