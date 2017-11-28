# Multi Layer Perceptron

import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron:
	def __init__(self, features, targets, nhidden, bias):

		self.features, self.targets = self.shuffle_train_validation(features, targets)
		n = int(0.7*features.shape[0])
		data_dict = {'features_train': 	   (lambda x: self.features[:n]),
					 'targets_train':  	   (lambda x: self.targets[:n]),
			 		 'features_validation':(lambda x: self.features[n:]),
			 		 'targets_validation': (lambda x: self.targets[n:])}

		self.features_train	     = data_dict['features_train'](self.features)
		self.features_validation = data_dict['features_validation'](self.features)
		self.targets_train	     = data_dict['targets_train'](self.targets) 
		self.targets_validation  = data_dict['targets_validation'](self.targets)
		self.nhidden = nhidden
		self.bias = bias

		'''
		self.w12 = np.random.uniform(size = (self.features.shape[1], self.nhidden))
		self.w23 = np.random.uniform(size = (self.nhidden, self.nhidden))
		self.w34 = np.random.uniform(size = (self.nhidden, self.targets.shape[1]))

		self.wb1 = np.random.uniform(size = (1, self.nhidden))
		self.wb2 = np.random.uniform(size = (1, self.nhidden))
		self.wb3 = np.random.uniform(size = (1, self.targets.shape[1]))
		'''

		self.w12 = np.random.randn(self.features.shape[1], self.nhidden)
		self.w23 = np.random.randn(self.nhidden, self.nhidden)
		self.w34 = np.random.randn(self.nhidden, self.targets.shape[1])

		self.wb1 = np.random.randn(1, self.nhidden)
		self.wb2 = np.random.randn(1, self.nhidden)
		self.wb3 = np.random.randn(1, self.targets.shape[1])
		


		funcs = {'sigmoid': (lambda x: 1/(1 + np.exp(-x)), lambda x: x*(1 - x)),
				 'linear' : (lambda x: x, lambda x: 1)}
		(self.activate,  self.activatePrime)  = funcs['sigmoid']
		(self.zactivate, self.zactivatePrime) = funcs['linear']

	def shuffle_train_validation(self, features, targets, flag = 'None'):
		idx = np.random.permutation(len(features))
		features, targets = features[idx], targets[idx]
		if flag == 'tv':
			return np.array([features[0]]), np.array([targets[0]])
		else:
			return features, targets

	def run(self, features):
		v0 = features
		y0 = v0
		y1 = self.activate(np.dot(v0, self.w12))  + self.bias*self.wb1
		y2 = self.activate(np.dot(y1, self.w23))  + self.bias*self.wb2
		z  = self.zactivate(np.dot(y2, self.w34)) + self.bias*self.wb3
		return z		

	def train(self, epochs, lr, alpha):
		mdw12, mdw23, mdw34 = np.zeros(self.w12.shape), np.zeros(self.w23.shape), np.zeros(self.w34.shape)
		mdwb1, mdwb2, mdwb3 = np.zeros(self.wb1.shape), np.zeros(self.wb2.shape), np.zeros(self.wb3.shape)
		Error_rms, Errorv_rms = [], []
		for e in range(epochs):

			# Trainning part
			v0, o = self.shuffle_train_validation(self.features_train, self.targets_train, 'tv')

			y0 = v0

			v1 = np.dot(y0, self.w12)  + self.bias*self.wb1
			y1 = self.activate(v1)

			v2 = np.dot(y1, self.w23)  + self.bias*self.wb2
			y2 = self.activate(v2)

			v3 = np.dot(y2, self.w34)  + self.bias*self.wb3
			z = self.zactivate(v3)

			error = z - o
			error_rms = 0.5*np.mean(error**2)

			gradk = error * self.zactivatePrime(z)
			gradj = np.dot(gradk, self.w34.T) * self.activatePrime(y2)
			gradi = np.dot(gradj, self.w23.T) * self.activatePrime(y1)

			dw34 = -gradk * y2.T
			dw23 = -gradj * y1.T
			dw12 = -gradi * v0.T

			self.w34 += (lr * dw34) + alpha*mdw34
			self.w23 += (lr * dw23) + alpha*mdw23
			self.w12 += (lr * dw12) + alpha*mdw12

			dwb3 = -gradk * self.bias
			dwb2 = -gradj * self.bias
			dwb1 = -gradi * self.bias

			self.wb3 += (lr * dwb3) + alpha*mdwb3
			self.wb2 += (lr * dwb2) + alpha*mdwb2
			self.wb1 += (lr * dwb1) + alpha*mdwb1

			mdw34, mdw23, mdw12 = dw34, dw23, dw12
			mdwb3, mdwb2, mdwb1 = dwb3, dwb2, dwb1

			Error_rms.append(error_rms)

			print('{:.1f}'.format(100*e/epochs))

			# Validation part
			v0, o = self.shuffle_train_validation(self.features_validation, self.targets_validation, 'tv')
			errorv = self.run(v0) - o
			errorv_rms = 0.5*np.mean(errorv**2)

			Errorv_rms.append(errorv_rms)

		return Error_rms, Errorv_rms




# Data for Neural Network

epochs = int(1e5)
lr = 1e-1
alpha = 1e-2
nhidden = 50
bias = 0

records = 100

features = np.array([[i/records] for i in range(records)])
targets = features*np.sin(features*np.pi*2) + 1

idx = np.random.permutation(features.shape[0])
features, targets = features[idx], targets[idx]

m = int(0.8*features.shape[0])
data_dic = {'features_train_validation': (lambda x: features[:m]),
		    'targets_train_validation':  (lambda x: targets[:m]),
	 		'features_test': 			 (lambda x: features[m:]),
	 		'targets_test':   			 (lambda x: targets[m:])}

features_train_validation = data_dic['features_train_validation'](features)
targets_train_validation  = data_dic['targets_train_validation'](targets)

features_test 			  = data_dic['features_test'](features)
targets_test		  	  = data_dic['targets_test'](targets)


nn = MultiLayerPerceptron(features_train_validation, targets_train_validation, nhidden, bias)
Error_rms, Errorv_rms = nn.train(epochs, lr, alpha)


plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), Error_rms,  label = 'train')
plt.plot(range(epochs), Errorv_rms, label = 'validation')
plt.ylim(0, 0.5)
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(features_test, nn.run(features_test), label = 'RNA output')
plt.scatter(features_test, targets_test,          label = 'Data test')
plt.legend()
plt.show()