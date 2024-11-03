import numpy as np
import matplotlib.pyplot as plt 

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


#a

#curve = C(t) & scalar field = f(x, y)

def f(x, y) :
    return x*(x + y**2)


#choose a sutiable intervall for t:

t = np.linspace(-2, 2, 20)

#this is the curve C(t)
x = t**2 - 2
y = t + 1

#function f on the curve
c = f(x,y)

figure = plt.figure(figsize = (7, 7))
ax = figure.add_subplot()

#scatter is used to plot the curve with colors to represent the scalar field values 
scalar_field = ax.scatter(x, y, c=c, cmap = 'coolwarm')
ax.plot(x, y, color = 'red')

#
print('x:', x, ' ,y: ', y, ' ,f(x, y): ', c)
#plt.colorbar(scalar_field, label=r'f(x,y)')
plt.grid(True)
plt.show()

#b
