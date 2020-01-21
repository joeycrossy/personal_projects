from base_class import Base
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import train_test_utils as tt_utils

class TrainTest(Base):
	def __init__(self, path_to_data):
		super().__init__(path_to_data)

	def train_model(self, model_name, **kwargs):
		self.model_name = model_name
		model_types = {
					"logistic": LogisticRegression,
					"rf": RandomForestClassifier,
					"xgb": XGBClassifier,
			}
		print(kwargs)
		if model_name in model_types:
			self.model = model_types[model_name](**kwargs)
			self.model.fit(self.X_train, self.y_train.values.ravel())
		
		elif model_name in ['tf', 'keras']:
			activation = 'relu'
			if "activation" in kwargs:
				activation = kwargs["activation"]
			self.model = tt_utils.get_tf_model(n_features = len(self.feature_names), 
												activation = activation)

			optimiser = 'adam'
			loss = 'binary_crossentropy'
			metrics = ['accuracy']

			if "optimiser" in kwargs:
				optimiser = kwargs['optimiser']
			if "loss" in kwargs:
				loss = kwargs['loss']
			if "metrics" in kwargs:
				metrics = kwargs['metrics']

			self.model.compile(optimizer=optimiser,
				              loss=loss,
				              metrics=metrics)

			epochs = 100
			batch_size = 32

			if "epochs" in kwargs:
				epochs = kwargs["epochs"]
			if "batch_size" in kwargs:
				batch_size = batch_size["batch_size"]

			self.model.fit(self.X_train, self.y_train.values.ravel(), 
						epochs = epochs, batch_size = batch_size)


	def test_model(self):
		self.train_pred = self.model.predict(self.X_train)
		self.val_pred = self.model.predict(self.X_val)

		self.cm_train = tt_utils.get_confusion_matrix(self.y_train.values.ravel(), self.train_pred)
		self.cm_val = tt_utils.get_confusion_matrix(self.y_val.values.ravel(), self.val_pred)

		if self.model_name not in ['tf', 'keras']:
			print(f"Train Score: {self.model.score(self.X_train, self.y_train):0.3f}")
			print(f"Val.  Score: {self.model.score(self.X_val, self.y_val):0.3f}")
		else:
			print(f"Train Score: {self.model.evaluate(self.X_train, self.y_train)}")
			print(f"Val.  Score: {self.model.evaluate(self.X_val, self.y_val)}")

	def save_model(self, file_name):
		with open(filename, 'wb') as f:
			pickle.dump(self.model, f)
