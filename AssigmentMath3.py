import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D #used for 3D scalar fields 

# #exercise  1

# #a)

# fig = pt.figure(figsize =(10, 10))
# ax = fig.add_subplot()


# #define the parameters 
# t = np.linspace(-2, 2, 100)
# x = t**5
# y = 4 - t**3

# ax.plot(x, y, label='A parametric curve')
# ax.legend()
# plt.show()


# #Exercise 6 Vector fields

# #a) 

# x = np.arange(-1, 1, 0.1)
# y = np.arange(-1, 1, 0.1)

# x, y = np.meshgrid(x, y)

# #this is the vector field 
# u = x + y 
# v = x - y

# figure = plt.figure(figsize = (7, 7))
# ax = figure.add_subplot()

# #this is used to plot the vector fields 
# ax.quiver(x, y ,u ,v, color = 'blue', scale = 10, headwidth = 2.5)

# #the interval for plot 
# ax.axis([-1, 1, -1, 1])
# plt.show()

# #b)

# x = np.arange(-1, 1, 0.1)
# y = np.arange(-1, 1, 0.1)

# x, y = np.meshgrid(x, y)

# #this is the vector field 
# u = x - y**2 
# v = x**2*y

# figure = plt.figure(figsize = (7, 7))
# ax = figure.add_subplot()

# #this is used to plot the vector fields 
# ax.quiver(x, y ,u ,v, color = 'green', scale = 10, headwidth = 2.5)

# #the interval for plot 
# ax.axis([-1, 1, -1, 1])
# plt.show()

# #c

# x = np.arange(-1, 1, 0.1)
# y = np.arange(-1, 1, 0.1)

# x, y = np.meshgrid(x, y)

# #this is the vector field 
# u = np.sin(x)*np.cos(y)
# v = np.sin(y) + np.cos(x*y**2)

# figure = plt.figure(figsize = (7, 7))
# ax = figure.add_subplot()

# #this is used to plot the vector fields 
# ax.quiver(x, y ,u ,v, color = 'pink', scale = 10, headwidth = 2.5)

# #the interval for plot 
# ax.axis([-1, 1, -1, 1])
# plt.curve(True)
# plt.show()

#Exercise 8 Vector fields on curves and surfaces 


# #a

# #curve = C(t) & scalar field = f(x, y)

# def f(x, y) :
#     return x*(x + y**2)


# #choose a sutiable intervall for t:

# t = np.linspace(-2, 2, 100)

# #this is the curve C(t)
# x = t**2 - 2
# y = t + 1

# #we plot f along the curve C 
# # f(c(t)) = (t**2 - 2)*((t**2 - 2) + (t + 1)**2)

# #function f on the curve
# fdotC = f(x,y) #this is the f(c(t)) or f(x(t), y(t))

# #figure = plt.figure(figsize = (7, 7))
# figure, ax = plt.subplots(figsize = (7, 7))

# #scatter is used to plot the curve with colors to represent the scalar field values 
# scalar_field = ax.scatter(x, y, c=fdotc, cmap = 'coolwarm', label = 'The scalar field f on the curve C')
# ax.plot(x, y, color = 'red', label = 'The curve C')
# #cbar = plt.colorbar(scalar_field, ax = ax, label = 'f on C')

# #
# print('x:', x, ' ,y: ', y, ' ,f(x, y): ', scalarV) #debugging
# plt.grid(True)
# plt.show()

# #b

# #the function 
# def f(x, y, z) :
#     return x**2*z*y

# #the choosen interval for t
# t = np.linspace(-3, 3, 20)

# #the curve values 
# x = t
# y = t**3 - 4*t + 1
# z = 1 - t**2

# #now combine the scalar field and the curve 
# fdotc = f(x, y, z)

# figure = plt.figure(figsize = (7, 7)) 
# ax = figure.add_subplot(111, projection='3d') # 111 is a shorthand for specofying the position of the subplot

# #make det scalar field 
# scalar_field = ax.scatter(x, y, z,  c = fdotc, cmap = 'viridis', label = 'The scalar field f on the curve C')
# ax.plot(x, y, z, color = 'blue', linewidth = 0.5, label = 'The curve C')

# plt.grid()
# plt.show()

# #c

# def f(x, y, z) :
#     return x**3*z + 3*y*z**2 - 3*x*z


# t = np.linspace(0, 20, 100)


# x = np.cos(t)
# y = np.sin(t)*np.cos(3*t)
# z = t*np.cos(2*t)

# #f(c(t))
# fdotc = f(z, y, z)

# figure = plt.figure(figsize = (7, 7)) 
# ax = figure.add_subplot(111, projection='3d') 

# scalar_field = ax.scatter(x, y, z,  c = fdotc, cmap = 'viridis', label = 'The scalar field f on the curve C')
# ax.plot(x, y, z, color = 'blue', linewidth = 0.5, label = 'The curve C')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.legend

# plt.grid()
# plt.show()



