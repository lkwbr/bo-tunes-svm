# Luke Weber, 11398889
# CptS 570, HW #6
# Created 12/16/16

"""
Contains both project init code and objective function used by Spearmint

Using Bayesian optimization library (Spearmint) on SVM classifier (LibSVM)
for automated hyper-parameter tuning!
"""

# Imports
import sys
sys.path.append("libsvm/libsvm-3.21/python")
from svmutil import *
from parse import *

# Globals
prob = None
validation_labels = None
validation_data = None
train_labels = None
train_data = None

def init():
	"""
	Train SVM on only first fold of Optical Character Recognition (OCR) data
	"""
	
	# Parse train data: [80%] train.txt, [20%] validation.txt
	global prob, validation_labels, validation_data, train_labels, train_data

	parse_input("data/ocr_fold0_sm_train.txt")
	train_labels, train_data = svm_read_problem("train.txt")
	validation_labels, validation_data = svm_read_problem("validation.txt")

	# Formulate SVM problem
    	prob = svm_problem(train_labels, train_data)

def main(job_id, params):
	"""
	Objective function called by Spearmint to determine desirability of 
	the parameters C and gamma:
		- Setup params
		- Train SVM, get model
		- Test SVM
		- Return accuracy
	"""

	# Setup
	init()

	# Set SVM params; NOTE: "-q" silences training output
	c = params["c"][0]
	g = params["g"][0]
	
	param = svm_parameter("-c {0} -g {1} -q".format(c, g))
	print("[JOB{0}] Training with c = {1:.3f}, g = {2:.3f}...".format(job_id, c, g))	

	# Train model
	m = svm_train(prob, param)

	# Predict
	pred_lbl, pred_acc, pred_val = svm_predict(validation_labels, validation_data, m)

	# Parse and return accuracy - inverse it so that
	# minimizing this function will maximize our accuracy
	acc = pred_acc[0]
	print("[LUKE_TOKEN] Accuracy = {0}".format(acc))	
	inv_acc = 100 - acc
	
	return inv_acc 
