from __future__ import print_function
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lab_c
import lasagne
from argparse import ArgumentParser


# Permute images
def permute_mnist(window_size, X_train, y_train, X_val, y_val, X_test, y_test):
	num_permute = window_size*window_size
	shift = (X_train.shape[1] - num_permute)/2
	perm_inds = range(num_permute)
	np.random.shuffle(perm_inds)
	perm_inds = np.array(perm_inds) +  shift

	def permute_one(inds, X):
		X_new = np.transpose(np.array([X[:,c] for c in inds]))  ### needs to check
		return X_new

	perm_inds = np.concatenate([np.arange(shift), perm_inds, np.arange(shift)+num_permute+shift])
	perm_inds = perm_inds.tolist()


	X_train_new = permute_one(perm_inds, X_train)
	X_val_new = permute_one(perm_inds, X_val)
	X_test_new = permute_one(perm_inds, X_test)

	return X_train_new, y_train, X_val_new, y_val, X_test_new, y_test


# Download and prepare the MNIST dataset
def load_dataset():
	# We first define a download function, supporting both Python 2 and 3.
	if sys.version_info[0] == 2:
		from urllib import urlretrieve
	else:
		from urllib.request import urlretrieve

	def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
		print("Downloading %s" % filename)
		urlretrieve(source + filename, filename)

	# We then define functions for loading MNIST images and labels.
	# For convenience, they also download the requested files if needed.
	import gzip

	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
		data = data.reshape(-1,784)

		return data / np.float32(256)

	def load_mnist_labels(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the labels in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=8)
		# The labels are vectors of integers now, that's exactly what we want.
		return data

	# We can now download and read the training and test set images and labels.
	X_train = load_mnist_images('train-images-idx3-ubyte.gz')
	y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
	X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
	y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

	# We reserve the last 10000 training examples for validation.
	X_train, X_val = X_train[:-10000], X_train[-10000:]
	y_train, y_val = y_train[:-10000], y_train[-10000:]

	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test


# Build Network
def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
					 drop_hidden=.5):

	network = lasagne.layers.InputLayer(shape=(None, 784),
										input_var=input_var)
	if drop_input:
		network = lasagne.layers.dropout(network, p=drop_input)
	# Hidden layers and dropout:
	nonlin = lasagne.nonlinearities.rectify
	for _ in range(depth):
		network = lasagne.layers.DenseLayer(
				network, width, nonlinearity=nonlin)
		if drop_hidden:
			network = lasagne.layers.dropout(network, p=drop_hidden)
	# Output layer:
	softmax = lasagne.nonlinearities.softmax
	network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
	return network


# Build Sparse Network
def build_add_mlp(mask, weight, input_var=None, depth=2, width=800, drop_input=.2,
					 drop_hidden=.5):

	network = lasagne.layers.InputLayer(shape=(None, 784),
										input_var=input_var)
	if drop_input:
		network = lasagne.layers.dropout(network, p=drop_input)
	# Hidden layers and dropout:
	nonlin = lasagne.nonlinearities.rectify
	i=0
	for _ in range(depth):
		network = lab_c.DenseLayer(mask[i+i], weight[i+i], network, width, nonlinearity=nonlin)
		if drop_hidden:
			network = lasagne.layers.dropout(network, p=drop_hidden)
		i = i +1
	# Output layer:
	softmax = lasagne.nonlinearities.softmax
	network = lab_c.DenseLayer(mask[i+i], weight[i+i], network, 10, nonlinearity=softmax)
	return network


# Batch iterator
def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


def degree_cal(mask, layer_size):

	degree = []
	for i in range(len(layer_size)-1): # i=0,1,2
		data_current = mask[i+i]
		if i < 1:
			first_degree = np.sum(data_current,axis = 1).astype(int)
			first_degree = first_degree[np.nonzero(first_degree)]
			# degree = degree + first_degree.tolist()
			degree.append(first_degree)

		# hidden nodes 
		current_degree = np.sum(data_current,axis=0).astype(int)  

		if i < len(layer_size)-2: # i=0,1
			data_next = mask[i+i+2]
			next_degree = np.sum(data_next,axis=1).astype(int)
			current_degree = current_degree+next_degree

		degree.append(current_degree)
	return degree


# generate random mask
def random_connection(mask,  connect_fraction, layer_size, layer_size_new):

	mask_new = []
	for i in range(len(layer_size)-1):
		W_mask = np.zeros((layer_size_new[i], layer_size_new[i+1])).astype('int16')
		b_mask = np.ones(layer_size_new[i+1]).astype('int16') ### all the bs are relearned
		mask_new.append(W_mask)
		mask_new.append(b_mask)

	for i in range(len(layer_size)-1): #i=0,1,2 the last laer is all relearned
		# the last layer
		if i > len(layer_size)-3:  
			mask_new[i+i][:,:] = 1

		else:
			add_connection = int(connect_fraction*layer_size[i]*layer_size[i+1])
			probs = np.outer(np.ones(layer_size[i]), np.ones(layer_size[i+1]))
			probs = (probs*(1-mask[i+i])).astype('float32') ### those already connected have 0 prob to be connected again
			probs = probs/probs.sum() ### normalize such that the summation of all entries included is 1.
			ind = np.random.choice(int(probs.size), size = (add_connection, 1), p=probs.flatten(), replace=False)
			mask_flatten  = mask_new[i+i].flatten()
			mask_flatten[ind] = 1
			mask_new[i+i] = mask_flatten.reshape(mask[i+i].shape)

	return mask_new


# generate new mask according to preferential attachment
def prefer_connection(mask,  connect_fraction, layer_size, layer_size_new):
	# width = layer_size[1]  
	mask_new = []
	for i in range(len(layer_size)-1):
		W_mask = np.zeros((layer_size_new[i], layer_size_new[i+1])).astype('int16')
		b_mask = np.ones(layer_size_new[i+1]).astype('int16') ### all the bs are relearned
		mask_new.append(W_mask)
		mask_new.append(b_mask)

	# get degrees for all nodes
	degree = degree_cal(mask, layer_size)

	for i in range(len(layer_size)-1): #i=0,1,2 the last layer is all relearned
		# the last layer
		if i > len(layer_size)-3:  
			mask_new[i+i][:,:] = 1
		else:
			add_connection = int(connect_fraction*layer_size[i]*layer_size[i+1])
			probs = np.outer(degree[i], degree[i+1])
			probs = (probs*(1-mask[i+i])).astype('float32') # those already connected have 0 prob to be connected again
			probs = probs/probs.sum() # normalize such that the summation of all entries included is 1.
			ind = np.random.choice(int(probs.size), size = (add_connection, 1), p=probs.flatten(), replace=False)
			mask_flatten  = mask_new[i+i].flatten()
			mask_flatten[ind] = 1
			mask_new[i+i] = mask_flatten.reshape(mask[i+i].shape)

	return mask_new


def dense_sparse(mask,  connect_fraction, layer_size, layer_size_new, param_values_cont):
	add_fraction = connect_fraction
	thres = []
	mask_new = []
	for i in range(len(param_values_cont)):
		data_current = np.abs(param_values_cont[i]*(1-mask[i]))
		if len(param_values_cont[i].shape)>1 and i < len(param_values_cont)-3:  ### the last layer is dense
			vec_data = data_current.flatten()
			tmp = int((1. - add_fraction)*data_current.size)  #only the dense part will be 1, all the pruned part will be 0
			thres_current = np.sort(vec_data)[tmp]
		else:
			thres_current = np.float32(0.0)

		mask_current = (data_current>=thres_current).astype('int16')
		thres.append(thres_current)
		mask_new.append(mask_current)

	return mask_new


# Main program
def main(permute_size, sparsity, model='mlp', num_epochs=200, method='AoB', connect_fraction=0.1):
	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
	X_train_new, y_train_new, X_val_new, y_val_new, X_test_new, y_test_new = permute_mnist(
		permute_size, X_train, y_train, X_val, y_val, X_test, y_test)
	input_var = T.fmatrix('inputs')
	target_var = T.ivector('targets')

	# Create neural network model (depending on first command line parameter)
	depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
	with np.load('model/sparse_model_{0}_{1}_{2}.npz'.format(sparsity, depth, width)) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]

	mask = []
	for i in range(len(param_values)):
		mask_current= np.array(param_values[i]!=0.).astype(int) ### dense part 1, sparse part 0
		mask.append(mask_current)

	wid = np.int(width)
	# add_nodes = np.int(add_fraction*wid)
	layer_size = [784, wid, wid , 10]
	layer_size_new = layer_size

	if method=='AoB':
		mask_new = random_connection(mask, connect_fraction, layer_size, layer_size_new)
	elif method=='ApB': 
		mask_new = prefer_connection(mask, connect_fraction, layer_size, layer_size_new)
	else:
		with np.load('cont_model/cont_dense_fc_{0}_{1}_{2}_{3}.npz'.format(sparsity, permute_size,depth, width)) as f:
			param_values_cont = [f['arr_%d' % i] for i in range(len(f.files))]
		mask_new = dense_sparse(mask, connect_fraction, layer_size, layer_size_new, param_values_cont)

	### add one some nodes, make the weight matrices bigger and assign submatrices
	### to retrain the last layer, set to 0 first, all entries are relearned
	param_values[-1]= np.zeros(param_values[-1].shape).astype('float32')
	param_values[-2]= np.zeros(param_values[-2].shape).astype('float32')

	param_values_aug=[]   ## used to constrcut the network
	for i in range(len(layer_size)-1):
		W_current = np.zeros((layer_size_new[i], layer_size_new[i+1])).astype('float32')
		W_current[0:layer_size[i],0:layer_size[i+1]] = param_values[i+i]
		b_current = np.zeros((layer_size_new[i+1])).astype('float32')
		b_current[0:layer_size[i+1]] = param_values[i+i+1]
		param_values_aug.append(W_current)
		param_values_aug.append(b_current)


	# rebuild sparse model
	network_c_s = build_add_mlp(mask_new, param_values_aug, input_var, int(depth), wid,
								   float(drop_in), float(drop_hid))

	# Create a loss expression for training
	prediction = lasagne.layers.get_output(network_c_s)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
					  dtype=theano.config.floatX)

	# Create update expressions for training
	params = lasagne.layers.get_all_params(network_c_s, trainable=True)
	W_grads = lab_c.compute_grads1(loss, network_c_s, mask_new)
	updates = lasagne.updates.nesterov_momentum(
			loss_or_grads=W_grads, params=params, learning_rate=0.01, momentum=0.9)
	# Create a loss expression for validation/testing.
	test_prediction = lasagne.layers.get_output(network_c_s, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
					  dtype=theano.config.floatX)


	train_fn = theano.function([input_var, target_var], [loss, acc, W_grads[0], W_grads[2], W_grads[4]], updates=updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
	# import ipdb; ipdb.set_trace()
	lasagne.layers.set_all_param_values(network_c_s, param_values_aug)


	# retraining
	print("Starting retraining...")
	best_val_acc = 0
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_loss = 0
		train_acc=0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train_new, y_train_new, 500, shuffle=True):
			inputs, targets = batch
			loss, acc, g1, g2, g3 = train_fn(inputs, targets)
			train_loss += loss
			train_acc += acc
			train_batches += 1

		train_loss = train_loss / train_batches
		train_acc = train_acc/train_batches*100

		# And a full pass over the validation data:
		val_loss = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val_new, y_val_new, 500, shuffle=False):
			inputs, targets = batch
			loss, acc = val_fn(inputs, targets)
			val_loss += loss
			val_acc += acc
			val_batches += 1

		val_loss = val_loss / val_batches
		val_acc = val_acc / val_batches * 100
		# Then we print the results for this epoch:

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			# After training, we compute and print the test error:
			test_loss = 0
			test_acc = 0
			test_batches = 0
			for batch in iterate_minibatches(X_test_new, y_test_new, 500, shuffle=False):
				inputs, targets = batch
				loss, acc = val_fn(inputs, targets)
				test_loss += loss
				test_acc += acc
				test_batches += 1
			test_loss = test_loss / test_batches
			test_acc = test_acc / test_batches * 100

		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_loss))
		print("  train accuracy:\t\t{:.2f} %".format(train_acc))
		print("  validation loss:\t\t{:.6f}".format(val_loss))
		print("  validation accuracy:\t\t{:.2f} %".format(val_acc))
		print("  test loss:\t\t\t{:.6f}".format(test_loss))
		print("  test accuracy:\t\t{:.2f} %".format(test_acc))

		result_folder = "prefer_model_pathnet/{0}".format(connect_fraction)
		if not os.path.exists(result_folder):
			os.mkdir(result_folder)
		with open("prefer_model_pathnet/{0}/cont_sparse_fc_{1}_{2}_{3}_{4}_{5}.txt".format(
				connect_fraction, method, sparsity, permute_size, depth, width), "a") as myfile:
			myfile.write("{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f} {6:.3f}\n".format(epoch, 
				train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--model",  type=str, dest="model",
				default='custom_mlp:2,400,0.0,0.0', help="model options")
	parser.add_argument("--permute_size",  type=int, dest="permute_size",
				default=8, help="permutation size")  
	parser.add_argument("--sparsity",  type=float, dest="sparsity",
				default=0.9, help="sparsity of sparse model")  
	parser.add_argument("--num_epochs",  type=int, dest="num_epochs",
				default=200, help="number of epochs")  
	# parser.add_argument("--add_fraction",  type=float, dest="add_fraction",
				# default=0.1, help="fraction of weights pruned away") 
	parser.add_argument("--method",  type=str, dest="method",
				default='AoB', help="methods: AoB,  ApB, ds")  
	# parser.add_argument("--add_nodes",  type=int, dest="add_nodes",
	# 			default=40, help="number of nodes added to each hiddn layer") 
	parser.add_argument("--connect_fraction",  type=float, dest="connect_fraction",
				default=0.1, help="fraction of added connections")  
	args = parser.parse_args()

	main(**vars(args))
