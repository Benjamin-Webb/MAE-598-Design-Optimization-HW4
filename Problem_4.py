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
	h[0] = (1.0/4.0)*x[0]**2 + (1.0/5.0)*x[1]**2 + (1.0/25.0)*x[2]**2 - 1.0
	h[1] = x[0] + x[1] - x[2]

	return h

def NewtonRalphson(x):
	# Newton-Ralphson method for non-linear system of eqs

	h = contraints(x)
	dhds = np.zeros((2, 2), dtype=np.single)
	S_new = np.zeros((2, 1), dtype=np.single)
	S_old = np.zeros((2, 1), dtype=np.single)
	j = np.uint8(1)
	S_new[0] = x[0]
	S_new[1] = x[1]
	S_old[0] = x[0]
	S_old[1] = x[1]
	new_x = x

	while np.linalg.norm(h) > 0.001 and j < 100:

		# Calculate gradient w.r.t. s at current step
		dhds[0, 0] = 0.5 * S_new[0]
		dhds[0, 1] = 0.4 * S_new[1]
		dhds[1, 0] = 1.0
		dhds[1, 1] = 1.0

		# Update solution
		S_new = S_old - np.linalg.inv(dhds) @ h

		# Update constraints
		new_x = np.vstack((S_new, x[2]))
		h = contraints(new_x)

		# Update old solution
		S_old = S_new

		# Update iteration Counter
		j += 1

	return new_x

def redGrad(x):
	# Calculate reduced gradient
	# x: 3x1 vector

	pf_pd = np.asarray(2.0*x[2], dtype=np.single).reshape(1, 1)
	pf_ps = np.asarray([2.0*x[0], 2.0*x[1]], dtype=np.single).reshape(1, 2)
	ph_ps = np.linalg.inv(np.asarray([[0.5*x[0], 0.4*x[1]], [1.0, 1.0]], dtype=np.single)).reshape(2, 2)
	ph_pd = np.asarray([[0.08*x[2]], [-1.0]], dtype=np.single).reshape(2, 1)

	return pf_pd - pf_ps @ ph_ps @ ph_pd

def linesearch(x, dfdd):
	# perform linesearch

	# Define parameters
	alpha = 1.0
	b = 0.5
	t = 0.3
	f = np.zeros((100, 1), dtype=np.single)
	phi = np.zeros((100, 1), dtype=np.single)
	j = np.uint8(0)
	f[0] = minFun(x)
	ph_ps = np.linalg.inv(np.asarray([[0.5 * x[0], 0.4 * x[1]], [1.0, 1.0]]))
	ph_pd = np.asarray([[0.08 * x[2]], [-1.0]])

	while f[j] > phi[j] and j < 100:

		# Update iteration counter
		j += 1

		# Determine inputs for f(alpha)
		dk_step = x[2] - alpha*dfdd
		sk_step = x[0:2].reshape(2, 1) + alpha*(ph_ps@ph_pd@dfdd)

		f[j] = minFun(np.vstack((sk_step, dk_step)))

		# Determine phi(alpha)
		phi[j] = minFun(x) - alpha * t * dfdd**2

		# Update alpha
		if f[j] > phi[j]:
			alpha = b*alpha

	return alpha

# Main progam code
if __name__ == '__main__':

	# Find initial point
	x = np.zeros((3, 1000), dtype=np.single)
	x[0, 0] = 1.0
	x[1, 0] = -1.0

	# Initialize/starting point
	x[:, 0] = NewtonRalphson(x[:, 0]).reshape(-1, )

	# Calculate initial reduced gradient
	dfdd = redGrad(x[:, 0])

	# Begin GRG loop
	dk_new = np.single(0)
	dk_old = x[2, 0]
	Sk_new = np.single(0)
	Sk_old = x[0:2, 0]
	k = np.uint16(0)
	while np.linalg.norm(dfdd, ord=2) > 0.001:

		# Determine alpha
		alpha = linesearch(x[:, k], dfdd)

		# Take step in decision space
		dk_new = dk_old - alpha*dfdd

		# Take linear step in state space
		ph_ps = np.linalg.inv(np.asarray([[0.5*x[0, k], 0.4*x[1, k]], [1.0, 1.0]], dtype=np.single))
		ph_pd = np.array([[0.08*x[2, k]], [-1.0]], dtype=np.single)
		Sk_new = Sk_old.reshape(2, 1) + alpha*(ph_ps@ph_pd*dfdd)

		# Making non-linear correction to Sk, returns updated X
		x[0:3, k+1] = NewtonRalphson(np.vstack((Sk_new, dk_new))).reshape(-1, )

		# Update GRG and iteration counter
		dfdd = redGrad(x[0:3, k+1])

		dk_old = dk_new
		Sk_old = Sk_new

		k += 1

	print('x1: %.3f' % (x[0, k]))
	print('x2: %.3f' % (x[1, k]))
	print('x3: %.3f' % (x[2, k]))
