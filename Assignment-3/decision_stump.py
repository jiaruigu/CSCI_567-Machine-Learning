import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		##################################################
		# TODO: implement "predict"
		##################################################
		res = []
		for x in features:
			res.append(self.s if x[self.d]>self.b else -self.s)
		return res