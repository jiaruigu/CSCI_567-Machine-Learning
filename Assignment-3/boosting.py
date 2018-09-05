import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
		sum = np.zeros(len(features))
		for t in range(self.T):
			sum += self.betas[t]*np.array(self.clfs_picked[t].predict(features))
		sum[sum>=0] = 1
		sum[sum<0] = -1
		return sum.tolist()
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		w_t = np.ones(N)* 1/N
		for t in range(self.T):
			epsilon_t = 999999
			for clf in self.clfs:
				temp = 0
				for n in range(N):
					temp += w_t[n]*(1 if labels[n]!=clf.predict([features[n]])[0] else 0)
				if temp<epsilon_t:
					epsilon_t = temp
					h_t = clf
			self.betas.append(1/2*np.log((1-epsilon_t)/epsilon_t))
			self.clfs_picked.append(h_t)
			sum_w = 0
			for n in range(N):
				w_t[n] = w_t[n]*np.e**(-self.betas[t]) if labels[n] == h_t.predict([features[n]])[0] else w_t[n]*np.e**(self.betas[t])
				sum_w += w_t[n]
			w_t = w_t/sum_w

		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		pi_t = np.ones(N)*0.5
		z_t = np.zeros(N)
		w_t = np.zeros(N)
		for t in range(self.T):
			for n in range(N):
				z_t[n] = ((labels[n]+1)/2-pi_t[n])/(pi_t[n]*(1-pi_t[n]))
				w_t[n] = pi_t[n]*(1-pi_t[n])
			epsilon_t = 999999
			for clf in self.clfs:
				temp = 0
				for n in range(N):
					temp += w_t[n]*(z_t[n]-clf.predict([features[n]])[0])**2
				if temp<epsilon_t:
					epsilon_t = temp
					h_t = clf
			self.clfs_picked.append(h_t)
			self.betas.append(0.5)
			for n in range(N):
				temp = 0
				for clf in self.clfs_picked:
					temp += 1/2*clf.predict([features[n]])[0]
				pi_t[n] = 1/(1+np.e**(-2*temp))
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	