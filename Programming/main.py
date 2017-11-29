# Multi Layer Perceptron

import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron:
	def __init__(self, features, targets, nhidden, data, init_weights):

		self.features, self.targets = self.shuffle_train_validation(features, targets)
		self.nhidden = nhidden


		if data == 'xor':
			self.features_train	     = self.features
			self.features_validation = self.features
			self.targets_train	     = self.targets
			self.targets_validation  = self.targets

		elif data == 'function':
			n = int(0.7*features.shape[0])
			data_dict = {'features_train': 	   (lambda x: self.features[:n]),
						 'targets_train':  	   (lambda x: self.targets[:n]),
				 		 'features_validation':(lambda x: self.features[n:]),
				 		 'targets_validation': (lambda x: self.targets[n:])}

			self.features_train	     = data_dict['features_train'](self.features)
			self.features_validation = data_dict['features_validation'](self.features)
			self.targets_train	     = data_dict['targets_train'](self.targets) 
			self.targets_validation  = data_dict['targets_validation'](self.targets)
			

		if init_weights == 'uniform':
			self.w12 = np.random.uniform(size = (self.features.shape[1], self.nhidden))
			self.w23 = np.random.uniform(size = (self.nhidden, self.nhidden))
			self.w34 = np.random.uniform(size = (self.nhidden, self.targets.shape[1]))
		
		elif init_weights == 'randn':
			self.w12 = np.random.randn(self.features.shape[1], self.nhidden)
			self.w23 = np.random.randn(self.nhidden, self.nhidden)
			self.w34 = np.random.randn(self.nhidden, self.targets.shape[1])		


		funcs = {'sigmoid': (lambda x: 1/(1 + np.exp(-x)), lambda x: x*(1 - x)),
				 'linear' : (lambda x: x, lambda x: 1)}
		(self.activate,  self.activatePrime)  = funcs['sigmoid']
		(self.zactivate, self.zactivatePrime) = funcs['linear']

	def shuffle_train_validation(self, features, targets, flag = 'None'):
		idx = np.random.permutation(features.shape[0])
		features, targets = features[idx], targets[idx]
		if flag == 'tv':
			return np.array([features[0]]), np.array([targets[0]])
		else:
			return features, targets

	def run(self, features):
		v0 = features
		y0 = v0
		y1 = self.activate(np.dot(v0, self.w12))
		y2 = self.activate(np.dot(y1, self.w23))
		z  = self.zactivate(np.dot(y2, self.w34))
		return z		

	def train(self, epochs, lr):
		alpha = lr*1e-1
		mdw12, mdw23, mdw34 = np.zeros(self.w12.shape), np.zeros(self.w23.shape), np.zeros(self.w34.shape)
		Error_rms, Errorv_rms = [], []
		for e in range(epochs):

			# Trainning part
			v0, o = self.shuffle_train_validation(self.features_train, self.targets_train, 'tv')

			y0 = v0

			v1 = np.dot(y0, self.w12)
			y1 = self.activate(v1)

			v2 = np.dot(y1, self.w23)
			y2 = self.activate(v2)

			v3 = np.dot(y2, self.w34)
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

			mdw34, mdw23, mdw12 = dw34, dw23, dw12

			Error_rms.append(error_rms)

			print('{:.1f}'.format(100*e/epochs))

			# Validation part
			v0, o = self.shuffle_train_validation(self.features_validation, self.targets_validation, 'tv')
			errorv = self.run(v0) - o
			errorv_rms = 0.5*np.mean(errorv**2)

			Errorv_rms.append(errorv_rms)

		return Error_rms, Errorv_rms




# Hyperparameters

epochs = int(2e5)
lr = 1e-1
nhidden = 10



# Kind of data

data, init_weights = 'function', 'uniform'


if data == 'xor':
	features = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
	targets = np.array([[0], [1], [1], [0]])

	nn = MultiLayerPerceptron(features, targets, nhidden, data, init_weights)

elif data == 'function':
	records = 100
	features = np.array([[i/records] for i in range(records)])
	targets = features*np.sin(features*np.pi*2)

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

	nn = MultiLayerPerceptron(features_train_validation, targets_train_validation, nhidden, data, init_weights)



# Trainning data

Error_rms, Errorv_rms = nn.train(epochs, lr)



# Plotting data

if data == 'xor':

	plt.figure(figsize = (5, 2.5))
	plt.plot(range(epochs), Error_rms,  label = 'train')
	plt.plot(range(epochs), Errorv_rms, label = 'validation')
	plt.ylim(0, 0.5)
	plt.legend()
	plt.show()

	print('\n')
	print(nn.run(features))

elif data == 'function':

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