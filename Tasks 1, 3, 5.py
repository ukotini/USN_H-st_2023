import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#, 3, 4, 5, 6, 8, 9, 13(abe), 14, 16, 17(ac), 19, 20, 23, 25, 26, 28, 31.

#Exercise 1

# #a)

# figure = plt.figure(figsize = (10, 10))
# ax = figure.add_subplot()

# t = np.linspace(-2, 2)
# x = t**5
# y = 4 - t**3

# ax.plot(x, y, label= 'A parametric curve')
# ax.legend()
# plt.grid(True)
# plt.show()


# #b)

# figure = plt.figure(figsize = (10, 10))
# ax = figure.add_subplot()

# theta = np.linspace(0, 2*np.pi) #while i was coding i thought that i could make a function here, uff
# x = 2*np.cos(theta)
# y = 5*np.sin(theta)

# ax.plot(x, y, label= 'A parametric curve')
# ax.legend()
# plt.grid(True)
# plt.show()

# #c

# t = np.linspace(-2, 2)
# x = t**5
# y = 1 - t**2
# z = t**3 + 4*t + 1

# figure = plt.figure(figsize = (10, 10)) 
# ax = figure.add_subplot(111, projection='3d') #prøve den i pdf men fungerte ikke så googla litt og fant denne :)

# ax.plot(x, y, z, label= 'A 3D parametric curve')
# ax.legend()
# plt.grid(True)
# plt.show()

# #d

# theta = np.linspace(0, 2*np.pi)
# x = np.cos(theta)
# y = theta*np.sin(theta)
# z = 3*np.cos(theta)*np.sin(5*theta)

# figure = plt.figure(figsize = (10, 10)) 
# ax = figure.add_subplot(111, projection='3d') 

# ax.plot(x, y, z, label= 'A 3D parametric curve')
# ax.legend()
# plt.grid(True)
# plt.show()

#Exercise 3

# #a

# def f(x, y) :
#     return x + 3*y - 1

# x = np.linspace(-1, 1, 40)
# y = np.linspace(-1, 1, 40)

# x, y = np.meshgrid(x, y) #this is the [-1, 1] x [-1, 1]

# X = x
# Y = y
# Z = f(X, Y)

# figure = plt.figure(figsize = (10, 10))
# ax = figure.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor='none')
# plt.show()

# #b The monkey saddle (wierd name, but okay)


# def f(x, y) :
#     return x**3 - 3*x*y**2

# x = np.linspace(-1, 1, 40)
# y = np.linspace(-1, 1, 40)

# x, y = np.meshgrid(x, y) #this is the [-1, 1] x [-1, 1]

# X = x
# Y = y
# Z = f(X, Y)

# figure = plt.figure(figsize = (10, 10))
# ax = figure.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor='none')
# plt.show()

# #c

# x = np.linspace(-1, 1, 40)
# y = np.linspace(-1, 1, 40)

# x, y = np.meshgrid(x, y) #this is the [-1, 1] x [-1, 1]

# X = x**3
# Y = 1 - y**3
# Z = x*y

# figure = plt.figure(figsize = (10, 10))
# ax = figure.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor='none')
# plt.show()

#d

# x = np.linspace(-2, 2, 40)
# y = np.linspace(-2, 2, 40)

# x, y = np.meshgrid(x, y) #this is the [-1, 1] x [-1, 1]

# X = x*y
# Y = y**2 - x**2
# Z = x + y

# figure = plt.figure(figsize = (10, 10))
# ax = figure.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor='none')
# plt.show()

# #e

# def f(x, y) :
#     return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# x = np.linspace(-6, 6, 40)
# y = np.linspace(-6, 6, 40)

# x, y = np.meshgrid(x, y) #this is the [-6, 6] x [-6, 6]

# X = x
# Y = y
# Z = f(X, Y)

# figure = plt.figure(figsize = (10, 10))
# ax = figure.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor='none')
# plt.show()

# #Exercise 4

# #Exercise 5

# #a

# def fun(x, y) :
#     return 1 - x**2 - y**3 - x + 3*y - x*y

# x = np.linspace(-4.0, 4.0, 100) #this is (x, y) ∈ [-4, 4]**2
# y = np.linspace(-4.0, 4.0, 100)

# u, v = np.meshgrid(x, y)
# w = fun(u, v)


# figure = plt.figure(figsize =(7, 7))
# ax = figure.add_subplot()

# levels = np.arange(-5, 5, 1) #this is the interval for a, 1 is the number of curves 

# cont = ax.contour(u, v, w, levels, colors = 'black')
# plt.show()

#b

# def fun(x, y) :
#     return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# x = np.linspace(-6.0, 6.0, 100) 
# y = np.linspace(-6.0, 6.0, 100)

# u, v = np.meshgrid(x, y)
# w = fun(u, v)


# figure = plt.figure(figsize =(7, 7))
# ax = figure.add_subplot()

# levels = np.arange(-5, 5, 1) #this is the interval for a, 1 is the number of curves 

# cont = ax.contour(u, v, w, levels, colors = 'black')
# plt.show()

#c

def fun(x, y) :
    return np.cos(2*x)*np.sin(y)

x = np.linspace(-2*np.pi, 2*np.pi, 100) 
y = np.linspace(-2*np.pi, 2*np.pi, 100)

u, v = np.meshgrid(x, y)
w = fun(u, v)


figure = plt.figure(figsize =(7, 7))
ax = figure.add_subplot()

levels = np.arange(-2, 2, 0.1) #(-5, 5, 1) ga bare en mesh aktig bilde så endret litt på dette
# endringer 1 -> 0.5 -> 0.2 -> 0.1 

cont = ax.contour(u, v, w, levels, colors = 'black')
plt.show()
