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

# plt.grid(True)
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

# plt.legend()

# plt.grid(True)
# plt.show()

# #Exercise 9 

# #a

# #the scalar field 
# def f(x, y, z) :
#     return x*y*z

# #define the intervals for x and y for the surface
# x = np.linspace(-1, 1, 50)
# y = np.linspace(-1, 1, 50)

# #a mesh grid for the x and y

# xmesh, ymesh = np.meshgrid(x, y)

# #define the surface
# X = 1 - xmesh**2 - ymesh**2 #paraboloid

# #f(X(x, y))
# fdotX = f(xmesh, ymesh, X)

# #plot the surface 
# figure = plt.figure(figsize=(10, 10))
# ax = figure.add_subplot(111, projection='3d')

# #now plot the surface of the paraboloid 
# ax.scatter(xmesh, ymesh, X, c = fdotX, cmap = 'viridis', label = 'Paraboloid', s = 10)
# #ax.plot_surface(xmesh, ymesh, X, color = 'blue', linewidth = 0.5, label = 'Paraoloid')
# #you can choose if you want to see a smooth surface or a dotted surface

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.legend()

# plt.grid(True)
# plt.show()

# #b

# def f(x, y, z) :
#     return (x + y + z)*np.e**(-x*y*z)

# Ix = np.linspace(-1, 1, 50)
# Iy = np.linspace(-1, 1, 50)

# x, y = np.meshgrid(Ix, Iy)

# X = x*y

# fdotX = f(x, y, X)

# figure = plt.figure(figsize=(10, 10))
# ax = figure.add_subplot(111, projection='3d')

# #now plot the surface of the paraboloid 
# ax.scatter(x, y, X, c = fdotX, cmap = 'viridis', label = 'surface X', s = 10)
# #ax.plot_surface(xmesh, ymesh, X, color = 'blue', linewidth = 0.5, label = 'Paraoloid')
# #you can choose if you want to see a smooth surface or a dotted surface

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.legend()

# plt.grid(True)
# plt.show()

# #c

# def f(x, y, z) :
#     return (x + y + z) / (x*y*x)

# Ix = np.linspace(-1, 1, 20)
# Iy= np.linspace(-1, 1, 20)

# x, y = np.meshgrid(Ix, Iy)

# P = x + x*y
# Q = x**2 + (1-x)*y
# R = -x + x*y

# fdotX = f(P, R, Q)

# figure = plt.figure(figsize=(10, 10))
# ax = figure.add_subplot(111, projection='3d')

# ax.scatter(P, Q, R, c = fdotX, cmap = 'viridis', label = 'surface X', s = 10)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.legend()

# plt.grid(True)
# plt.show()

#Exercise 13

#a Riemann 

# define the r(t) = (t**2, t)
def r(t) :
    x = t**2
    y = t
    return x, y

# define the dericative of r(t)

def dr(t) : 
    dx = 2*t
    dy = 1
    return dx, dy

# define the vector field field u(x, y) = (y, x)
def u(x, y) :
    u1 = y
    u2 = x
    return u1, u2

# this is the intergrand fot the line integral 
def integrand(t) :
    x, y = r(t)             # getting the coordinates of the curve r(t)
    u1, u2 = u(x, y)        # evaluating the vector field
    dx, dy = dr(t)          # getting the tangent vector 
    return  u1 * dx + u2*dy #computing the dot product 


def upperRiemannSum(a, b, n) :
    tvalue = np.linspace(a, b, n)
    dt = (b - a)/n
    upperBound = 0

    for i in range(1, n) :
        upperBound += integrand(tvalue[i]) * dt
    return upperBound

def lowerRiemannSum(a, b, n) :
    tvalue = np.linspace(a, b, n)
    dt = (b - a)/n
    lowerBound = 0

    for i in range(n - 1) :
        lowerBound += integrand(tvalue[i]) * dt
    return lowerBound

# def midpointRiemannSum(a, b, n):
#     dt = (b - a) / n
#     midpointBound = 0

#     for i in range(n):
#         t_mid = a + (i + 0.5) * dt  # Calculate the midpoint of each subinterval
#         midpointBound += integrand(t_mid) * dt

#     return midpointBound

def intergrateRiemann(a, b, n, epsilon) :
    upperSum = upperRiemannSum(a, b, n)
    lowerSum = lowerRiemannSum(a, b, n)
    # midpointSum = midpointRiemannSum(a, b, n)
    difference = abs(upperSum - lowerSum)

    if difference < epsilon :
        average = (upperSum + lowerSum)/2
        print('I = {} +- {}' .format(average, difference))
        return True 
    else :
        print('Difference > epsilon')
        print('Upper: ', upperSum, ' Lower: ', lowerSum)
        # print('Midpoint:', midpointSum)
        print('Try higher n-value')
        return False

# parameters for use
a, b = -1, 2    # the interval for t
n = 1100        # number of subintervals 
epsilon = 0.1  # chosen tolarance 

resultRiemann = intergrateRiemann(a, b, n, epsilon)

if resultRiemann is not None : 
    print('Riemann sum: ', resultRiemann)
    
#a Simpson

# define the r(t) = (t**2, t)
def r(t) :
    x = t**2
    y = t
    return x, y

# define the dericative of r(t)

def dr(t) : 
    dx = 2*t
    dy = 1
    return dx, dy

# define the vector field field u(x, y) = (y, x)
def u(x, y) :
    u1 = y
    u2 = x
    return u1, u2

# this is the intergrand fot the line integral 
def integrand(t) :
    x, y = r(t)             # getting the coordinates of the curve r(t)
    u1, u2 = u(x, y)        # evaluating the vector field
    dx, dy = dr(t)          # getting the tangent vector 
    return  u1 * dx + u2*dy #computing the dot product 


def simpson(a, b, n) : 
    assert n % 2 == 0, 'n is not even'

    epsilon = (b - a)/n
    tvalues = np.linspace(a, b, n + 1)
    
    #defining the g-vector as an array 
    lst = [1] 
    for i in range(1, n) :
        if i % 2 == 0 :
            lst.append(2)
        else:
            lst.append(4)
    lst.append(1)

    #g-vector converted into an array
    gvec = np.array(lst, dtype = int)

    #computing f_0, ... , f_n
    fvec = np.array([integrand(t) for t in tvalues])

    #scalar product (dot product of the two vectors)
    integral = (epsilon / 3) * np.dot(gvec, fvec)

    #error 
    #error = -epsilon**4/180

    return integral # , error

# parameters for use
a, b = -1, 2    # the interval for t
n = 1100         # number of subintervals 
# epsilon = 0.01  # chosen tolarance 

resultSimpson = (simpson(a, b, n))
print('Simpson: ', resultSimpson)

#hell

        









