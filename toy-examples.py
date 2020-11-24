# -*- coding: utf-8 -*-
"""
Optimization algorithms in python
"""
import numpy as np
import matplotlib.pyplot as plt

#input: define a cost function with minimum (4.9, 2.3)
def cost_fn(x, y):
    return (x-4.9)**2 + (y-2.3)**2

### uniform search:
#input: define intervals to search over
intervals = np.arange(11)
#rest of the code is runs regardless of inputs
num_intervals = len(intervals)

points = np.zeros((num_intervals**2,2))
points[:,0] = np.repeat(intervals, num_intervals)
points[:,1] = np.tile(intervals, num_intervals)

fn_values = cost_fn(points[:,0], points[:,1])
fn_min = fn_values.min()
solution = points[fn_values == fn_min,]

### random search:
#input: decide on the region to search over + number of points to generate
num_points, start_point, end_point = 121, 0, 10
#rest of the code runs regardless of inputs
points = np.random.uniform(start_point, end_point, num_points * 2)
points = points.reshape(num_points, 2)

fn_values = cost_fn(points[:,0], points[:,1])
fn_min = fn_values.min()
solution = points[fn_values == fn_min,]

### gradient descent
def cost_fn_derivative(x,y):
    output = np.zeros(2)
    output[0] = 2 * (x - 4.9)
    output[1] = 2 * (y - 2.3)
    return output

start_point = np.array([0,0])
num_iterations = 100
step_length = 0.1

points = np.zeros((num_iterations, 2))
fn_values = np.zeros(num_iterations)
points[0,:] = start_point
grad = np.zeros(2)
for i in range(num_iterations):
    if(i > 0): 
        points[i] = points[i-1]- grad * step_length
    fn_values[i] = cost_fn(points[i, 0], points[i, 1])
    grad = cost_fn_derivative(points[i, 0], points[i, 1])

plt.plot(range(num_iterations), fn_values)
