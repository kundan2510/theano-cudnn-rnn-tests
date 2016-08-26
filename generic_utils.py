import numpy as np
import os
import theano

def floatX(num):
	if theano.config.floatX == 'float32':
		return np.float32(num)
	else:
		raise Exception("{} type not supported".format(theano.config.floatX))

def txt_to_list(txt_file, delimiter='\n'):
	file_ = open(txt_file, 'rb')
	file_list_raw = file_.read().split('\n')
	file_.close()
	file_list = []

	for sample in file_list_raw:
		if os.path.isfile(sample):
			file_list.append(sample)

	assert(len(file_list) > 0), "No kidding!! I can't create stream of batches from an empty list"

	return file_list

def create_folder_if_not_there(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)
		print "Created folder {}".format(folder)

def rolling_window(a, window_size, overlap = 0):
	# TODO: its buggy, remove the bug
	strides = a.strides[:-1]

	sz = a.strides[-1]

	shape = a.shape[:-1] + ( (a.shape[-1] - window_size +1)// (window_size -overlap), window_size)

	strides += (sz*(window_size -overlap), sz)

	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window_no_overlap(a, window_size):
	strides = a.strides[:-1]

	sz = a.strides[-1]

	shape = a.shape[:-1] + ( (a.shape[-1])// window_size, window_size)

	strides += (sz*window_size, sz)

	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def print_args(args):
	print "\nFollowing args will be used: \n"
	for k in args.__dict__.keys():
		print "{0:{fill}{align}30}: {1}".format(k, args.__dict__[k], fill=" ", align="<")
