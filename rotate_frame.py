import numpy as np

# https://en.wikipedia.org/wiki/Rotation_matrix

# Function to rotate frame by some theta
def rotate(theta, x0, y0, x, y):
    
	newx = (x-x0)*np.cos(theta)-(y-y0)*np.sin(theta)

	newy = (x-x0)*np.sin(theta)+(y-y0)*np.cos(theta)

	return newx+x0, newy+y0

# Function to shift frame by some dx and dy
def shift(dx,dy,x,y):
    
	newx = x+dx
	newy = y+dy

	return newx, newy

# Find offset in both rotation and trasposition
def offset(x,y,dx,dy,theta,x0,y0):
    
	newx_shift, newy_shift = shift(dx,dy,x,y)
    
	newx, newy = rotate(theta, x0, y0, newx_shift, newy_shift)

	return newx, newy

# Function that returns the sum of the offsets
def vect_diff_sum(x, gaia_x, gaia_y, offset_x, offset_y, x0, y0):

	'''
	Params
	------
	x - list
		[dx, dy, theta, x0, y0] where dx/dy is a shift in the x/y-direction, theta is a rotation angle, and x0/y0 is the rotation center
	gaia_x - arr
		Gaia x-positions that we assume to be the true x-positions
	gaia_y - arr
		Gaia y-positions that we assume to be the true x-positions
	offset_x - arr
		The incorrect x-positions of objects
	offset_y - arr
		The incorrect y-positions of objects

	Outputs
	--------
	diff_sum_new - float
		The sum of the differences after applying some arbitrary offset correction. We want to minimize this value in order to find the true offset.
	'''

	dx,dy,theta = x

	new_x, new_y = offset(offset_x,offset_y,dx,dy,theta,x0,y0)

	x_diff = gaia_x-new_x
	y_diff = gaia_y-new_y
	vector_diff_new = np.sqrt(x_diff**2+y_diff**2)
	diff_sum_new = np.sum(vector_diff_new)

	return diff_sum_new