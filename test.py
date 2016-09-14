from __future__ import absolute_import, print_function

from layers import GRU, WrapperLayer
from models import Model
import theano
import theano.tensor as T
import numpy
from theano.gpuarray import dnn

EPS = 1e-6

def check_equality_two_nd_array(a, b):
	squared_difference = numpy.sum((a-b)**2)

	return squared_difference < EPS

def test_gru(depth, input_dim, hidden_dim):
	'''hidden_dim and output_dim are usually same'''

	model = Model() # To collect parameters and keep track of layers

	X = T.tensor3('X') # input
	h0 = T.tensor3('h0') # initial hidden state of recurrent nets

	last_layer = WrapperLayer(X)
	last_dim = input_dim
	for i in range(depth):
		gru = GRU(last_dim, hidden_dim, last_layer,  name = "layer_{}".format(i+1), s0 = h0[i,:,:])
		model.add_layer(gru)
		last_layer = gru
		last_dim = hidden_dim

	params = model.get_params()
	print("Printing order of params. Important to know as this will help set params for cudnn_rnn")

	model.print_params()

	#list_of_param_values = [p.get_value() for p in params] #list of param values

	#output = last_layer.output() # output tensor

	#forward_fun = theano.function([X, h0], output) #forward function


	#Y = T.tensor3('Y') # proxy tensor with which we want to match the output of rnn to get a loss


	'''For checking gradient, I am defining loss as following,
	 here 'output' is the theano tensor representing output of rnn op'''

	#loss = T.mean((Y - output)*(Y - output)) # mean square error

	#grad = T.grad(loss, params) # list of gradient with respect to parameters

	#get_grad = theano.function([X, h0, Y], grad) # getting list of gradients
	rnnb = dnn.RNNBlock('float32', hidden_dim, depth, 'gru')
	psize = rnnb.get_param_size([2, input_dim])
	params = theano.shared(numpy.zeros((psize,), dtype='float32'))
	# irm, irb, ium, iub, inm, inb, rrm, rrb, rum, rub, rnm, rnb
	l0params = rnnb.split_params(params, 0, [2, input_dim])
	print([p.shape for p in l0params])

	import sys;sys.exit(0)
	'''
	loss_rnn = T.mean((Y-output_cudnn)*(Y - output_cudnn))
	grad_cudnn = T.grad(loss, params_cudnn)
	'''




	"""
	cudnn_rnn_forward_fun = theano.function([X], cudnn_rnn_gru_output)

	"""

	def test0(bs, ts):
		'''
		bs: batch_size
		ts: number of timesteps
		'''
		h0 = numpy.random.random((depth, bs, hidden_dim)).astype('float32')
		inp1 = numpy.random.random((bs, ts, input_dim)).astype('float32')
		out1 = forward_fun(inp1, h0)
		'''checking output shape'''
		assert(out1.shape == (bs, ts, hidden_dim))
		'''
		out1_cudnn = cudnn_rnn_forward_fun(out1)
		assert(out1_cudnn.shape == (10, 3, hidden_dim))
		'''

		'''
		# Checking Values

		assert(check_equality_two_nd_array(out1, out1_cudnn))

		'''

	def test1(bs, ts):
		'''
		bs: batch_size
		ts: number of timesteps
		'''
		inp1 = numpy.random.random((bs, ts, input_dim)).astype('float32')
		h0 = numpy.random.random((depth, bs, hidden_dim)).astype('float32')
		Y = numpy.random.random((bs,ts,hidden_dim)).astype('float32')

		grad1 = get_grad(inp1, h0, Y)

		'''
		grad_cudnn = get_grad_cudnn(inp1, h0, Y)
		'''

		'''
			compare grad with cudnn_grad here
		'''

		'''
		for g, g_hat in zip(grad1, grad_cudnn):
			check_equality_two_nd_array(g, g_hat)
		'''

	test0(10, 5)
	print("passed test0 -1")
	test0(1, 10)
	print("passed test0 -2")

	test1(5, 3)
	print("passed test1 -1")

	test1(6, 10)
	print("passed test1 -2")


print("Running Case - 1")
test_gru(1, 2, 3)

print("Running Case - 2")
test_gru(3, 5, 6)

print("Running Case - 3")
test_gru(3, 1, 1)


print("Running Case - 4")
test_gru(3, 1, 20)

print("Running Case - 5")
test_gru(3, 20, 1)




