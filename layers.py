from __future__ import absolute_import, print_function

import theano
import theano.tensor as T
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.gpuarray.dnn import dnn_conv

from generic_utils import *
srng = RandomStreams(seed=3732)

T.nnet.relu = lambda x: T.switch(x > floatX(0.), x, floatX(0.))


def dropout(X, is_train, drop_prob=0.5):
	# is_train should be a theano scalar. 1 means training and 0 means validation/testing
	retain_prob = floatX(1 - drop_prob)
	dropped_X = X*srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
	dropped_X /= retain_prob
	return theano.ifelse.ifelse(T.eq(is_train, floatX(1.0)), dropped_X, X)

# def batch_normalisation(input, gamma, beta, is_train, sample_mean, sample_var):


def uniform(stdev, size):
    """uniform distribution with the given stdev and size"""
    return numpy.random.uniform(
        low=-stdev * numpy.sqrt(3),
        high=stdev * numpy.sqrt(3),
        size=size
    ).astype(theano.config.floatX)


def linear_transform_weights(input_dim, output_dim, initialization = 'glorot', param_list = None, name = "", w_normalization=True):
	"theano shared variable given input and output dimension and initialization method"
	if initialization == 'glorot':
		weight_inialization = uniform(numpy.sqrt(2.0/input_dim),(input_dim, output_dim))
	else:
		raise Exception("Not Implemented Error: {} initialization not implemented".format(initialization))

	W = theano.shared(weight_inialization, name=name)

	assert(param_list is not None)

	if w_normalization:
		norm_val = numpy.linalg.norm(weight_inialization, axis=0)
		g = theano.shared(norm_val, name=W.name+'.g')
		W_normed = W * (g / W.norm(2, axis=0)).dimshuffle('x',0)
		param_list.append(W)
		param_list.append(g)
		return W_normed
	else:
		param_list.append(W)
		return W

def bias_weights(length, initialization='zeros', param_list = None, name = ""):
	"theano shared variable for bias unit, given length and initialization"
	if initialization == 'zeros':
		bias_initialization = numpy.zeros(length).astype(theano.config.floatX)
	else:
		raise Exception("Not Implemented Error: {} initialization not implemented".format(initialization))

	bias =  theano.shared(
			bias_initialization,
			name=name
			)

	if param_list is not None:
		param_list.append(bias)

	return bias

def get_conv_2d_filter(filter_shape, subsample = (1,1), param_list = None, masktype = None, name = "", initialization='glorot'):
	if initialization == 'glorot':

		fan_in = numpy.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(subsample))
		w_std = numpy.sqrt(6.0 / (fan_in + fan_out))

		filter_init = uniform(w_std, filter_shape)

		assert(filter_shape[2] % 2 == 1), "Only filters with odd dimesnions are allowed."
		assert(filter_shape[3] % 2 == 1), "Only filters with odd dimesnions are allowed."

		if masktype is not None:
			filter_init *= floatX(2.0)

		conv_filter = theano.shared(filter_init, name = name)
		param_list.append(conv_filter)

		if masktype is not None:
			mask = numpy.zeros(
				filter_shape,
				dtype=theano.config.floatX
				)

			if filter_shape[3] == 1 and filter_shape[2] == 1:
				raise Exception("Masking not allowed for (1,1) filter shape")
			elif filter_shape[3] == 1:
				mask[:,:,:filter_shape[2]//2,:] = floatX(1.)
				if masktype == 'f':
					mask[:,:,filter_shape[2]//2,:] = floatX(1.)
			elif filter_shape[2] == 1:
				mask[:,:,:,:filter_shape[3]//2] = floatX(1.)
				if masktype == 'f':
					mask[:,:,:,:filter_shape[3]//2] = floatX(1.)
			else:
				center_row = filter_shape[2]//2
				centre_col = filter_shape[3]//2
				if masktype == 'f':
					mask[:,:,:center_row,:] = floatX(1.)
					mask[:,:,center_row,:centre_col+1] = floatX(1.)
				elif masktype == 'b':
					mask[:,:,:center_row,:] = floatX(1.)
					mask[:,:,center_row,:centre_col] = floatX(1.)
				elif masktype == 'p':
					mask[:,:,:center_row,:] = floatX(1.)

			conv_filter = conv_filter*mask

		return conv_filter
	else:
		raise Exception('Not Implemented Error')


class Layer:
	'''Generic Layer Template which all layers should inherit'''
	def __init__(name = ""):
		self.name = name
		self.params = []

	def get_params():
		return self.params

class GRU(Layer):
	def __init__(self, input_dim, output_dim, input_layer, is_train = None, drop_p = 0.0, s0 = None, batch_normalize = False, name="" ):
		'''Layers information'''
		self.name = name
		self.input_dim = input_dim
		self.hidden_dim = output_dim
		self.output_dim = output_dim
		self.input_layer = input_layer
		self.X = input_layer.output().dimshuffle(1,0,2)
		self.s0 = s0
		self.params = []

		'''Dropout applied on input'''
		if drop_p > 0.0:
			assert(is_train is not None)
			self.X = dropout(self.X,is_train,drop_p)

		'''Layers weights'''

		'''self.params is passed so that any paramters could be appended to it'''

		self.W_i = linear_transform_weights(input_dim, 3*output_dim, param_list=self.params, name=name+".W_i", w_normalization=False)
		self.b_i = bias_weights((3*output_dim, ), param_list=self.params, name=name+".b_i")

		self.W_s = linear_transform_weights(self.hidden_dim, 3*output_dim, param_list=self.params, name=name+".W_s", w_normalization=False)


		'''calculating processed input for all time steps in one go'''
		processed_input = T.dot(self.X, self.W_i) + self.b_i 

		'''step through processed input to create output'''
		def step(processed_input_curr, s_prev):
			processed_prev_state = T.dot(s_prev, self.W_s)

			gates = T.nnet.sigmoid(
				processed_prev_state[:,:2*self.hidden_dim] + \
				processed_input_curr[:,:2*self.hidden_dim]
				)

			update = gates[:,:self.hidden_dim]
			reset = gates[:,self.hidden_dim:]

			hidden = T.tanh(
				processed_input_curr[:,2*self.hidden_dim:] + \
				reset * processed_prev_state[:, 2*self.hidden_dim:]
				)
			

			s_curr = ((floatX(1) - update) * hidden) + (update * s_prev)

			return s_curr

		outputs_info = self.s0

		states, updates = theano.scan(
				fn=step,
				sequences=[processed_input],
				outputs_info = outputs_info
			)

		self.Y = states.dimshuffle(1,0,2)

	def output(self):
		return self.Y


class Embedding(Layer):
	'''docstring for Embedding'''
	def __init__(self, num_symbols, output_dim, input_matrix, name=""):
		self.name = name
		self.num_symbols = num_symbols
		self.output_dim = output_dim
		self.input_matrix = input_matrix
		self.emb_matrix = theano.shared(
			uniform(0.2, (num_symbols,output_dim)),
			name = name+".emb_matrix"
			)
		self.params = [self.emb_matrix]

	def output(self):
		return self.emb_matrix[self.input_matrix]

class Softmax(Layer):
	def __init__(self, input_layer,  name=""):
		self.input_layer = input_layer
		self.name = name
		self.params = []
		self.input_dim = input_layer.output_dim
		self.output_dim = input_layer.output_dim
		self.X = self.input_layer.output()
		self.input_shape = self.X.shape

	def output(self):
		return T.nnet.softmax(self.X.reshape((-1,self.input_shape[self.X.ndim-1]))).reshape(self.input_shape)

class FC(Layer):
	def __init__(self, input_dim, output_dim, input_layer, is_train = None, drop_p = 0.0, name=""):
		self.input_layer = input_layer
		self.name = name
		self.params = []
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.X = self.input_layer.output()

		'''Dropout applied on input'''
		if drop_p > 0.0:
			assert(is_train is not None)
			self.X = dropout(self.X,is_train,drop_p)

		self.W = linear_transform_weights(input_dim, output_dim, param_list = self.params, name=name+".W", w_normalization=True)
		self.b = bias_weights((output_dim,), param_list = self.params, name = name+".b")

	def output(self):
		return T.dot(self.X, self.W) + self.b


class Concat(Layer):
	def __init__(self, input_layers, axis = 1, name=""):
		self.input_layers = input_layers
		self.X = [l.output() for l in self.input_layers]
		self.name = name
		self.axis = axis
		self.params = []

	def output(self):
		return T.concatenate(self.X, axis=self.axis)

class Conv2D(Layer):
	"""
	input_shape: (batch_size, input_channels, height, width)
	"""
	def __init__(self, input_layer, input_channels, output_channels, filter_size, subsample = (1,1), border_mode='half', masktype = None, name = ""):
		self.X = input_layer.output()
		self.name = name
		self.subsample = subsample
		self.border_mode = border_mode

		self.params = []

		if isinstance(filter_size, tuple):
			self.filter_shape = (output_channels, input_channels, filter_size[0], filter_size[1])
		else:
			self.filter_shape = (output_channels, input_channels, filter_size, filter_size)

		self.filter = get_conv_2d_filter(self.filter_shape, param_list = self.params, initialization = 'glorot', masktype = masktype, name=name+'.filter')

		self.bias = bias_weights((output_channels,), param_list = self.params, name = name+'.b')

	def output(self):
		conv_out = dnn_conv( self.X, self.filter, border_mode = self.border_mode, conv_mode='cross', subsample=self.subsample)
		# conv_out = T.nnet.conv2d(self.X, self.filter, border_mode = self.border_mode, filter_flip=False)
		return conv_out + self.bias[None,:,None,None]



class Conv1D(Layer):
	"""
	input_shape : (batch_size, width, input_channels)
	"""
	def __init__(self, input_layer, input_channels, output_channels, filter_size, subsample = (1,1), border_mode='valid', masktype = None, name = ""):
		self.X = input_layer.output()
		self.name = name
		self.subsample = subsample
		self.border_mode = border_mode

		self.params = []

		self.filter_shape = (output_channels, input_channels, 1, filter_size)

		self.filter = get_conv_2d_filter(self.filter_shape, param_list = self.params, initialization = 'glorot', masktype = masktype, name=name+'.filter')

		self.bias = bias_weights((output_channels,), param_list = self.params, name = name+'.b')

	
	def apply(self, X):
		X = X.dimshuffle(0,2,'x',1)
		conv_out = T.nnet.conv2d(X, self.filter, border_mode = self.border_mode, filter_flip=False)
		temp_out =  conv_out + self.bias[None,:,None,None]
		temp_out =  temp_out.dimshuffle(0,3,1,2)
		return temp_out.reshape((temp_out.shape[0], temp_out.shape[1], -1))

	def output(self):
		return self.apply(self.X)


class WrapperLayer(Layer):
	def __init__(self, X, name=""):
		self.params = []
		self.name = name
		self.X = X

	def output(self):
		return self.X

