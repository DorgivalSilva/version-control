# Multi Layer Perceptron

import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron():
	def __init__(self, features, targets):
		self.features = features
		self.targets  = targets

		self.nhidden = int((self.features.shape[1] + self.targets.shape[1])/2) + 1

		self.w12 = 2*np.random.rand(self.features.shape[1], self.nhidden) - 1
		self.w23 = 2*np.random.rand(self.nhidden, self.nhidden)  		  - 1
		self.w34 = 2*np.random.rand(self.nhidden, self.targets.shape[1])  - 1

		activation = 'sigmoid'
		funcs = {'sigmoid': (lambda x: 1/(1 + np.exp(-x)), lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1 +np.exp(-x)))),
				 'linear':  (lambda x: x, lambda x: 1)}
		(self.activate, self.activatePrime) = funcs[activation]

	def run(self, features):
		h0 = features
		h1 = self.activate(np.dot(h0, self.w12))
		h2 = self.activate(np.dot(h1, self.w23))
		z  = self.activate(np.dot(h2, self.w34))
		return z		

	def train(self, features, targets, epochs, lr):
		Total_error = []
		for e in range(epochs):
			error_rms = 0
			dw12, dw23, dw34 = np.zeros(self.w12.shape), np.zeros(self.w23.shape), np.zeros(self.w34.shape)
			for x, y in zip(features, targets):

				h0 = x
				h1 = self.activate(np.dot(h0, self.w12))
				h2 = self.activate(np.dot(h1, self.w23))
				z  = self.activate(np.dot(h2, self.w34))

				error = z - y

				delError     = error*self.activatePrime(z)

				hidden23_error = np.dot(self.w34, delError)
				delHidden23 = hidden23_error*self.activatePrime(h2)

				hidden12_error = np.dot(self.w23, delHidden23)
				delHidden12 = hidden12_error*self.activatePrime(h1)

				dw34 += delError*h2[:,None]
				dw23 += delHidden23*h1[:,None]
				dw12 += delHidden12*x[:,None]

				error_rms += np.mean(error**2)

			self.w34 += lr*dw34/features.shape[0]
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