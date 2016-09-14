import theano
import theano.tensor as T
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda.dnn import dnn_conv

from generic_utils import *

def uniform(stdev, size):
    """uniform distribution with the given stdev and size"""
    return numpy.random.uniform(
        low=-stdev * numpy.sqrt(3),
        high=stdev * numpy.sqrt(3),
        size=size
    ).astype(theano.config.floatX)


def linear_transform_weights(input_dim, output_dim, initialization = 'glorot', param_list = None, name = "", w_normalization=False):
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

		'''Layers weights'''

		'''self.params is passed so that any paramters could be appended to it'''

		self.W_i = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name+".W_i", w_normalization=False)
		self.W_r = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name+".W_r", w_normalization=False)
		self.W_h = linear_transform_weights(input_dim, output_dim, param_list=self.params, name=name+".W_h", w_normalization=False)
		self.R_i = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name+".R_i", w_normalization=False)
		self.R_r = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name+".R_r", w_normalization=False)
		self.R_h = linear_transform_weights(output_dim, output_dim, param_list=self.params, name=name+".R_h", w_normalization=False)
		self.b_rh = bias_weights((output_dim, ), param_list=self.params, name=name+".b_rh")
		self.b_rr = bias_weights((output_dim, ), param_list=self.params, name=name+".b_rr")
		self.b_ru = bias_weights((output_dim, ), param_list=self.params, name=name+".b_ru")
		self.b_wi = bias_weights((output_dim, ), param_list=self.params, name=name+".b_wi")
		self.b_wr = bias_weights((output_dim, ), param_list=self.params, name=name+".b_wr")
		self.b_wh = bias_weights((output_dim, ), param_list=self.params, name=name+".b_wh")


		'''step through processed input to create output'''
		def step(inp, s_prev):
			i_t = T.nnet.sigmoid(
				T.dot(inp, self.W_i) + T.dot(self.R_i, s_prev) + self.b_wi + self.b_ru
				)
			r_t = T.nnet.sigmoid(
				T.dot(inp, self.W_r) + T.dot(self.R_r, s_prev) + self.b_wr + self.b_rr
				)

			h_hat_t = T.tanh(
				T.dot(inp, self.W_h) + (r_t*(T.dot(self.R_h, s_prev) + self.b_rh)) + self.b_wh
				)
			

			s_curr = ((floatX(1) - i_t) * h_hat_t) + (i_t * s_prev)

			return s_curr

		outputs_info = self.s0

		states, updates = theano.scan(
				fn=step,
				sequences=[self.X],
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


class WrapperLayer(Layer):
	def __init__(self, X, name=""):
		self.params = []
		self.name = name
		self.X = X

	def output(self):
		return self.X

