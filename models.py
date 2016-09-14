from __future__ import absolute_import, print_function

import pickle
import numpy

class Model:
	def __init__(self, name=""):
		self.name = name
		self.layers = []
		self.params = []
		self.other_updates = {}

	def add_layer(self,layer):
		self.layers.append(layer)
		for p in layer.params:
			self.params.append(p)

		if hasattr(layer, 'other_updates'):
			for y in layer.other_updates:
				self.other_updates[y[0]]=y[1]

	def print_layers(self):
		for layer in self.layers:
			print(layer.name)

	def get_params(self):
		return self.params

	def print_params(self):
		total_params = 0
		for p in self.params:
			curr_params = numpy.prod(numpy.shape(p.get_value()))
			total_params += curr_params
			print("{} ({})".format(p.name, curr_params))
		print("total number of parameters: {}".format(total_params))
		print("Note: Effective number of parameters might be less due if you are using masking!!")

	def combine_with(self, model, new_name):
		'''
		An utility function to combine another instance of Model class with self
		Please note that this will not change connections between nodes.
		'''
		self.name = new_name
		self.layers = self.layers + model.layers
		self.params = self.params + model.params

		for k in model.other_updates:
			self.other_updates[k] = model.other_updates[k]


	def save_params(self, file_name):
		params = {}
		for p in self.params:
			params[p.name] = p.get_value()
		pickle.dump(params, open(file_name, 'wb'))

	def load_params(self, file_name):
		params = pickle.load(open(file_name, 'rb'))
		for p in self.params:
			p.set_value(params[p.name])
