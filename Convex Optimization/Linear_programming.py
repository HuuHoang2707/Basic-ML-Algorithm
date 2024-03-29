from cvxopt import matrix, solvers

c = matrix([5., 10., 15., 4.])  # Two variables
A = matrix([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])  # Four constraints
b = matrix([600., 400.])  # Four constraints
G = matrix([[1., 0., -1., 0., 0., 0.], [1., 0., 0., -1., 0., 0.], [0., 1., 0., 0., -1., 0.], [0., 1., 0., 0., 0., -1.]])
h = matrix([800., 700., 0., 0., 0., 0.])

solvers.options['show_progress'] = False

sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
print("Solution ")
print(sol['x'])
