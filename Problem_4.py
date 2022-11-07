# MAE 598 - HW4
# Benjamin Webb
# 11/5/2022

import numpy as np
#import torch

def minFun(x):
	# Function to be minimized
	# x: 3x1 vector of input values
	# Returns f(x)

	return x[0]**2 + x[1]**2 + x[2]**2

def contraints(x):
	# Tensor of equality constraints
	# x: 3x1 vector of decision and state variabls
	# Returns h(x)

	h = np.zeros((2, 1), dtype=np.single)
	h[0] = 0.25*x[0]**2 + 0.2*x[1]**2 + 0.04*x[2]**2 - 1
	h[1] = x[0] + x[1] - x[2]

	return h

def NewtonRalphson(x):
	# Newton-Ralphson method for non-linear system of eqs

	h = contraints(x)
	dhds = np.zeros((2, 2), dtype=np.float)
	S = np.zeros((2, 100), dtype=np.float)
	j = np.uint8(1)
	S[0, 0] = x[0]
	S[1, 0] = x[1]

	while np.linalg.norm(h) > 0.001 and j < 100:

		# Calculate gradient w.r.t. s at current step
		dhds[0, 0] = 0.5*x[0]
		dhds[0, 1] = 0.4*x[1]
		dhds[1, 0] = 1.0
		dhds[1, 1] = 1.0

		# Update solution
		S[j] = S[j-1] - np.linalg.inv(dhds)*h

		# Update constraints
		h = contraints([[S[0, j]], [S[1, j]], x[2]])

		# Update iteration Counter
		j += 1

	return S
