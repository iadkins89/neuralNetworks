import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w_one = np.random.rand(X.shape[1])
        self.b_one = .5
        self.w_two = np.random.rand(1)
        self.b_two = .5

    def __sigmoid(self, x) : return 1/(1+np.exp(-1*x))

    def __d_sigmoid(self,x): return self.__sigmoid(x)*(1 - self.__sigmoid(x))

    def __relu(self, x):
        if x > 0: return x
        else: return 0

    def __d_relu(self, x):
        if x > 0: return 1
        else: return 0

    def __loss(self, o,y): return np.square(o-y)

    def __forward(self, x):
        z = x.dot(self.w_one) + self.b_one
        a = self.__sigmoid(z)
        o = a * self.w_two + self.b_two
        return z,a,o

    def __serialize(self, dep, x, z, a, o, opt ):
        serialize = self.__get_serializer(opt)
        return serialize(dep, x, z, a, o)

    def __get_serializer(self, opt):
        if opt == 'relu':
            return self.__backprop_relu
        elif opt == 'sigmoid':
            return self.__backprop_sigmoid
        else:
            raise ValueError(opt)
    def __backprop_sigmoid(self, dep, x, z, a, o):
        self.w_two = self.w_two - self.learn_rate * a * (o - dep)
        self.b_two = self.b_two - self.learn_rate * (o - dep)
        self.w_one = self.w_one - self.learn_rate * self.w_two.dot(o - dep) * self.__d_sigmoid(z)*x
        self.b_one = self.b_one - self.learn_rate * self.w_two.dot(o - dep) * self.__d_sigmoid(z)

    def __backprop_relu(self, dep, x, z, a, o):
        self.w_two = self.w_two - self.learn_rate * a * (o - dep)
        self.b_two = self.b_two - self.learn_rate * (o - dep)
        self.w_one = self.w_one - self.learn_rate * self.w_two.dot(o - dep) * self.__d_relu(z)*x
        self.b_one = self.b_one - self.learn_rate * self.w_two.dot(o - dep) * self.__d_relu(z)

    def train(self, num_epochs, learn_rate , option):
        self.learn_rate = learn_rate
        cost = []
        predictions =[]
        for epoch in range(0, num_epochs):
            for i in range(len(self.X)):
                z, a, o = self.__forward(self.X[i])
                self.__serialize(self.y[i],self.X[i], z, a, o, option) #back propagation
                cost.append(self.__loss(o, self.y[i]))

                if epoch == range(0, num_epochs)[-1]:
                    predictions.append(o)

            print(f'Epoch {epoch}| Acc {(1 - (sum(cost) / len(cost))) * 100} | Error {(sum(cost) / len(cost))}')

        return predictions, cost

    def predict(self, xt, yt):
        cost = []
        predictions = []
        for i in range(len(xt)):
            zt, at, ot = self.__forward(xt[i])
            cost.append(self.__loss(ot, yt[i]))
            predictions.append(ot)

        print(f'Acc {(1 - (sum(cost) / len(cost))) * 100} | Error {(sum(cost) / len(cost))}')

        return predictions
###################################################################

def normalize(x): return (x-np.mean(x))/np.std(x)
data_with_dates = pd.read_csv('data/Spotter_Cleaned.csv')
data = data_with_dates.iloc[:,1:]
data['Mean Period (s)'] = np.log(data['Mean Period (s)'])
data['Mean Period (s)'] = normalize(data['Mean Period (s)'])
data['Peak Direction (deg)'] = normalize(data['Peak Direction (deg)'])
dep = np.array(data['Significant Wave Height (m)'])
ind = np.array(data[['Peak Direction (deg)', 'Mean Period (s)']])

learn_rate = .01
num_epochs = 10

mlp = Perceptron(ind,dep)
p,c = mlp.train(num_epochs, learn_rate, 'sigmoid')


test_with_dates = pd.read_csv('data/NDBC_Cleaned.csv')
test_data = test_with_dates.iloc[:,1:]
test_data['APD'] = np.log(test_data['APD'])
test_data['APD'] = normalize(test_data['APD'])
test_data['MWD'] = normalize(test_data['MWD'])
test_dep = np.array(test_data['WVHT'])
test_ind = np.array(test_data[['MWD', 'APD']])

pt = mlp.predict(test_ind, test_dep)

fig, ax = plt.subplots(nrows= 2, ncols=1)
ax[0].plot(range(len(ind)), p,  linewidth=2.0, color='darksalmon' , label = ' Training Predictions')
ax[0].plot(range(len(ind)), dep, '--',  linewidth=1.0, color='darkcyan', label = 'Training Actual Values')
ax[0].legend(loc="upper left")

ax[1].plot(range(len(test_ind)), pt, color='darksalmon', linewidth=2.0, label = 'Predictions')
ax[1].plot(range(len(test_ind)), test_dep, '--', color='darkcyan', linewidth=1.0, label = 'Actual Value')
ax[1].legend(loc="upper left")
plt.show()



