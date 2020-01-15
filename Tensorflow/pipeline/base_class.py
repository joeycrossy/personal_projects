import pandas as pd

class Base():
	def __init__(self, path_to_data):
		self.datapath = path_to_data
		self.file_ending = self.datapath.split('.')[-1]

	def extract_data(self, **kwargs):
		self.kwargs = kwargs
		open_func_dict = {
						'csv': pd.read_csv, 
						'pkl': pd.read_pickle,
						'pickle': pd.read_pickle,
						'json': pd.read_json
						}
		if self.file_ending in open_func_dict:
			self.df = open_func_dict[self.file_ending](self.datapath, **self.kwargs)

		else:
			print(f"Cannot identify method to open file of type: {self.file_ending}")

	def prepare_data(self, target = 'target'):
		self.target = target
		self.feature_names = [i for i in self.df if i != self.target]

		self.X = self.df[self.feature_names]
		self.y = self.df[[self.target]]

	def engineer_features(self):
		pass

	def load_model(self):
		pass