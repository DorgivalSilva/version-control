# Multi Layer Perceptron

import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron:
	def __init__(self, features, targets):
		self.features = features
		self.targets  = targets

		self.nhidden = int((self.features.shape[1] + self.targets.shape[1])/2) + 1
		
		self.w12 = np.random.uniform(size = (self.features.shape[1], self.nhidden))
		self.w23 = np.random.uniform(size  =(self.nhidden, self.targets.shape[1]))

		funcs = {'sigmoid': (lambda x: 1/(1 + np.exp(-x)), lambda x: x*(1 - x)),
				 'linear' : (lambda x: x, lambda x: 1)}
		(self.activate,  self.activatePrime)  = funcs['sigmoid']
		(self.zactivate, self.zactivatePrime) = funcs['sigmoid']


	def run(self, features):
		v0 = features
		y1 = self.activate(np.dot(v0, self.w12))
		y2  = self.activate(np.dot(y1, self.w23))
		return y2		

	def shuffle_data(self, features, targets):
		idx = np.random.permutation(len(features))
		features, targets = features[idx], targets[idx]
		return np.array([features[0]]), np.array([targets[0]])

	def train(self, features, targets, epochs, lr):
		Error_rms = []
		for e in range(epochs):

			v0, o = self.shuffle_data(features, targets)

			v1 = np.dot(v0,  self.w12)
			y1 = self.activate(v1)

			v2 = np.dot(y1,  self.w23)
			y2 = self.zactivate(v2)

			error = y2 - o
			error_rms = 0.5*np.mean(error**2)

			gradk = error * self.zactivatePrime(y2)
			gradj = np.dot(gradk, self.w23.T) * self.activatePrime(y1)

			self.w23 += (lr * -gradk * y1.T)
			self.w12 += (lr * -gradj * v0.T)
		
			Error_rms.append(error_rms)

		return Error_rms

# Data for Neural Network

epochs = 100000
lr = 0.1

features = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
targets  = np.array([[0], [1], [1], [0]])

nn = MultiLayerPerceptron(features, targets)
Error_rms = nn.train(features, targets, epochs, lr)

print(nn.run(features))

plt.plot(range(epochs), Error_rms)
plt.show()