from base_class import Base
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import train_test_utils as tt_utils

class TrainTest(Base):
	def __init__(self, path_to_data):
		super().__init__(path_to_data)

	def train_model(self, model_name, **kwargs):
		model_types = {
					"rf": RandomForestClassifier,
					"xgb": XGBClassifier
			}
		print(kwargs)
		if model_name in model_types:
			self.model = model_types[model_name](**kwargs)
			self.model.fit(self.X_train, self.y_train.values.ravel())
		
		elif model_name in ['tf', 'keras']:
			self.model = tt_utils.get_tf_model()

			optimiser = 'adam'
			loss = 'binary_crossentropy'
			metrics = 'accuracy'

			# if optimiser in 

			self.model.compile(optimizer='adam',
				              loss='binary_crossentropy',
				              metrics=['accuracy'])
		


	def test_model(self):
		self.train_pred = self.model.predict(self.X_train)
		self.val_pred = self.model.predict(self.X_val)

		self.cm_train = tt_utils.get_confusion_matrix(self.y_train.values.ravel(), self.train_pred)
		self.cm_val = tt_utils.get_confusion_matrix(self.y_val.values.ravel(), self.val_pred)

		print(f"Train Score: {self.model.score(self.X_train, self.y_train):0.3f}")
		print(f"Val.  Score: {self.model.score(self.X_val, self.y_val):0.3f}")

	def save_model(self, file_name):
		with open(filename, 'wb') as f:
			pickle.dump(self.model, f)
