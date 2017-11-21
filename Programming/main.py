# Multi Layer Perceptron

import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron():
	def __init__(self, features, targets):
		self.features = features
		self.targets = targets

		self.nhidden = int((self.features.shape[1] + self.targets.shape[1])/2)

		self.w12 = 2*np.random.rand(self.features.shape[1], self.nhidden) - 1
		self.w23 = 2*np.random.rand(self.nhidden, self.targets.shape[1]) - 1

		activation = 'sigmoid'
		funcs = {'sigmoid': (lambda x: 1/(1 + np.exp(-x)), lambda x: (1/(1 + np.exp(-x))) * (1 - 1/(1 + np.exp(-x))))}
		(self.activate, self.activatePrime) = funcs[activation]

	def run(self, features):
		print(features)
		h1 = self.activate(np.dot(features, self.w12))
		z  = self.activate(np.dot(h1, self.w23))
		return z		

	def train(self, features, targets, epochs, lr):
		Total_error = []
		for e in range(epochs):
			dw12, dw23 = np.zeros(self.w12.shape), np.zeros(self.w23.shape)
			for x, y in zip(features, targets):

				h1 = self.activate(np.dot(x, self.w12))
				z  = self.activate(np.dot(h1, self.w23))

				error = z - y

				delError     = error*self.activatePrime(z)
				hidden_error = np.dot(self.w23, delError)
				delHidden    = hidden_error*self.activatePrime(h1)

				dw23 += delError*h1[:,None]
				dw12 += delHidden*x[:,None]

				error_rms = np.mean(error**2)

			self.w23 += lr*dw23/features.shape[0]
			self.w12 += lr*dw12/features.shape[0]

			Total_error.append(error_rms)

		return Total_error







# Data for Neural Network

epochs = 1000
lr = 0.4

features = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
targets  = np.array([[0], [1], [1], [0]])

nn = MultiLayerPerceptron(features, targets)

Total_error = nn.train(features, targets, epochs, lr)
print(nn.run(features))

plt.plot(range(epochs), Total_error)


