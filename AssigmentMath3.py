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

# #Exercise 13

# #a Riemann 

# # define the r(t) = (t**2, t)
# def r(t) :
#     x = t**2
#     y = t
#     return x, y

# # define the dericative of r(t)

# def dr(t) : 
#     dx = 2*t
#     dy = 1
#     return dx, dy

# # define the vector field field u(x, y) = (y, x)
# def u(x, y) :
#     u1 = y
#     u2 = x
#     return u1, u2

# # this is the intergrand fot the line integral 
# def integrand(t) :
#     x, y = r(t)             # getting the coordinates of the curve r(t)
#     u1, u2 = u(x, y)        # evaluating the vector field
#     dx, dy = dr(t)          # getting the tangent vector 
#     return  u1 * dx + u2*dy #computing the dot product 


# def upperRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     upperBound = 0

#     for i in range(1, n) :
#         upperBound += integrand(tvalue[i]) * dt
#     return upperBound

# def lowerRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     lowerBound = 0

#     for i in range(n - 1) :
#         lowerBound += integrand(tvalue[i]) * dt
#     return lowerBound


# def intergrateRiemann(a, b, n, epsilon) :
#     upperSum = upperRiemannSum(a, b, n)
#     lowerSum = lowerRiemannSum(a, b, n)
#     # midpointSum = midpointRiemannSum(a, b, n)
#     difference = abs(upperSum - lowerSum)

#     if difference < epsilon :
#         average = (upperSum + lowerSum)/2
#         print('I = {} +- {}' .format(average, difference))
#         return True 
#     else :
#         print('Difference > epsilon')
#         print('Upper: ', upperSum, ' Lower: ', lowerSum)
#         # print('Midpoint:', midpointSum)
#         print('Try higher n-value')
#         return False

# # parameters for use
# a, b = -1, 2    # the interval for t
# n = 1100        # number of subintervals 
# epsilon = 0.1  # chosen tolarance 

# resultRiemann = intergrateRiemann(a, b, n, epsilon)

# if resultRiemann is not None : 
#     print('Riemann sum: ', resultRiemann)
    
# #a Simpson

# # define the r(t) = (t**2, t)
# def r(t) :
#     x = t**2
#     y = t
#     return x, y

# # define the dericative of r(t)

# def dr(t) : 
#     dx = 2*t
#     dy = 1
#     return dx, dy

# # define the vector field field u(x, y) = (y, x)
# def u(x, y) :
#     u1 = y
#     u2 = x
#     return u1, u2

# # this is the intergrand fot the line integral 
# def integrand(t) :
#     x, y = r(t)             # getting the coordinates of the curve r(t)
#     u1, u2 = u(x, y)        # evaluating the vector field
#     dx, dy = dr(t)          # getting the tangent vector 
#     return  u1 * dx + u2*dy #computing the dot product 


# def simpson(a, b, n) : 
#     assert n % 2 == 0, 'n is not even'

#     epsilon = (b - a)/n
#     tvalues = np.linspace(a, b, n + 1)
    
#     #defining the g-vector as an array 
#     lst = [1] 
#     for i in range(1, n) :
#         if i % 2 == 0 :
#             lst.append(2)
#         else:
#             lst.append(4)
#     lst.append(1)

#     #g-vector converted into an array
#     gvec = np.array(lst, dtype = int)

#     #computing f_0, ... , f_n
#     fvec = np.array([integrand(t) for t in tvalues])

#     #scalar product (dot product of the two vectors)
#     integral = (epsilon / 3) * np.dot(gvec, fvec)

#     #error 
#     #error = -epsilon**4/180

#     return integral # , error

# # parameters for use
# a, b = -1, 2    # the interval for t
# n = 100         # number of subintervals 
# # epsilon = 0.01  # chosen tolarance 

# resultSimpson = (simpson(a, b, n))
# print('Simpson: ', resultSimpson)



# #b Riemann and Simpson together (they use the same code, writing the same function is redundant)

# # define the r(t) = (cos(t), sin(t)) unit cirle
# def r(t) :
#     x = np.cos(t)
#     y = np.sin(t)
#     return x, y

# # define the dericative of r(t)

# def dr(t) : 
#     dx = -np.sin(t)
#     dy = np.cos(t)
#     return dx, dy

# # define the vector field field u(x, y) = (cos(x), sin(y)**2)
# def u(x, y) :
#     u1 = np.cos(x)
#     u2 = np.sin(y)**2
#     return u1, u2

# # this is the intergrand fot the line integral 
# def integrand(t) :
#     x, y = r(t)             # getting the coordinates of the curve r(t)
#     u1, u2 = u(x, y)        # evaluating the vector field
#     dx, dy = dr(t)          # getting the tangent vector 
#     return  u1 * dx + u2*dy #computing the dot product 


# def upperRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     upperBound = 0

#     for i in range(1, n) :
#         upperBound += integrand(tvalue[i]) * dt
#     return upperBound

# def lowerRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     lowerBound = 0

#     for i in range(n - 1) :
#         lowerBound += integrand(tvalue[i]) * dt
#     return lowerBound


# def intergrateRiemann(a, b, n, epsilon) :
#     upperSum = upperRiemannSum(a, b, n)
#     lowerSum = lowerRiemannSum(a, b, n)
#     # midpointSum = midpointRiemannSum(a, b, n)
#     difference = abs(upperSum - lowerSum)

#     if difference < epsilon :
#         average = (upperSum + lowerSum)/2
#         print('I = {} +- {}' .format(average, difference))
#         return True 
#     else :
#         print('Difference > epsilon')
#         print('Upper: ', upperSum, ' Lower: ', lowerSum)
#         # print('Midpoint:', midpointSum)
#         print('Try higher n-value')
#         return False
    
# def simpson(a, b, n) : 
#     assert n % 2 == 0, 'n is not even'

#     epsilon = (b - a)/n
#     tvalues = np.linspace(a, b, n + 1)
    
#     #defining the g-vector as an array 
#     lst = [1] 
#     for i in range(1, n) :
#         if i % 2 == 0 :
#             lst.append(2)
#         else:
#             lst.append(4)
#     lst.append(1)

#     #g-vector converted into an array
#     gvec = np.array(lst, dtype = int)

#     #computing f_0, ... , f_n
#     fvec = np.array([integrand(t) for t in tvalues])

#     #scalar product (dot product of the two vectors)
#     integral = (epsilon / 3) * np.dot(gvec, fvec)

#     #error 
#     #error = -epsilon**4/180

#     return integral # , error

# # parameters for use
# a, b = 0, 2*np.pi    # the interval for t when we have a circle 
# n = 1000        # number of subintervals 
# epsilon = 0.01  # chosen tolarance 

# resultRiemann = intergrateRiemann(a, b, n, epsilon)

# if resultRiemann is not None : 
#     print('Riemann sum: ', resultRiemann)


# resultSimpson = (simpson(a, b, n))
# print('Simpson: ', resultSimpson)

# #c Riemann and Simpson

# # define the r(t) = (t, t**3, 1-2*t**4)
# def r(t) :
#     x = t
#     y = t**3
#     z = 1 - 2*t**4
#     return x, y, z

# # define the dericative of r(t)

# def dr(t) : 
#     dx = 1
#     dy = 3*t
#     dz = -8*t**3
#     return dx, dy, dz

# # define the vector field field u(x, y) = (x + yz, -yz, x**2 + y**2 + z**2)
# def u(x, y, z) :
#     u1 = x + y*z
#     u2 = -y*z
#     u3 = x**2 + y**2 + z**2
#     return u1, u2, u3

# # this is the intergrand fot the line integral 
# def integrand(t) :
#     x, y, z= r(t)             # getting the coordinates of the curve r(t)
#     u1, u2, u3 = u(x, y, z)        # evaluating the vector field
#     dx, dy, dz = dr(t)          # getting the tangent vector 
#     return  u1 * dx + u2*dy + u3*dz #computing the dot product 


# def upperRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     upperBound = 0

#     for i in range(1, n) :
#         upperBound += integrand(tvalue[i]) * dt
#     return upperBound

# def lowerRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     lowerBound = 0

#     for i in range(n - 1) :
#         lowerBound += integrand(tvalue[i]) * dt
#     return lowerBound


# def intergrateRiemann(a, b, n, epsilon) :
#     upperSum = upperRiemannSum(a, b, n)
#     lowerSum = lowerRiemannSum(a, b, n)
#     # midpointSum = midpointRiemannSum(a, b, n)
#     difference = abs(upperSum - lowerSum)

#     if difference < epsilon :
#         average = (upperSum + lowerSum)/2
#         print('I = {} +- {}' .format(average, difference))
#         return True 
#     else :
#         print('Difference > epsilon')
#         print('Upper: ', upperSum, ' Lower: ', lowerSum)
#         # print('Midpoint:', midpointSum)
#         print('Try higher n-value')
#         return False
    
# def simpson(a, b, n) : 
#     assert n % 2 == 0, 'n is not even'

#     epsilon = (b - a)/n
#     tvalues = np.linspace(a, b, n + 1)
    
#     #defining the g-vector as an array 
#     lst = [1] 
#     for i in range(1, n) :
#         if i % 2 == 0 :
#             lst.append(2)
#         else:
#             lst.append(4)
#     lst.append(1)

#     #g-vector converted into an array
#     gvec = np.array(lst, dtype = int)

#     #computing f_0, ... , f_n
#     fvec = np.array([integrand(t) for t in tvalues])

#     #scalar product (dot product of the two vectors)
#     integral = (epsilon / 3) * np.dot(gvec, fvec)

#     #error 
#     #error = -epsilon**4/180

#     return integral # , error

# # parameters for use
# a, b = -1, 1    # the interval for t (to see both the negative and posivite values of t)
# n = 1000        # number of subintervals 
# epsilon = 0.1  # chosen tolarance 

# resultRiemann = intergrateRiemann(a, b, n, epsilon)

# if resultRiemann is not None : 
#     print('Riemann sum: ', resultRiemann)


# resultSimpson = (simpson(a, b, n))
# print('Simpson: ', resultSimpson)


# Exercise 14

# def r(t) :
#     x = t**4
#     y = t**5 + t + 1
#     return x, y

# def dr(t) : 
#     dx = 4*t**3
#     dy = 5*t**4 + 1
#     return dx, dy

# def f(x, y) :
#     return x * y

# def integrand(t) :
#     x, y = r(t)  
#     dx, dy = dr(t)
#     return f(x, y) * np.sqrt(dx**2 + dy**2)  #2D case for the arc length          

# def upperRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     upperBound = 0

#     for i in range(1, n) :
#         upperBound += integrand(tvalue[i]) * dt
#     return upperBound

# def lowerRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     lowerBound = 0

#     for i in range(n - 1) :
#         lowerBound += integrand(tvalue[i]) * dt
#     return lowerBound


# def intergrateRiemann(a, b, n, epsilon) :
#     upperSum = upperRiemannSum(a, b, n)
#     lowerSum = lowerRiemannSum(a, b, n)

#     difference = abs(upperSum - lowerSum)

#     if difference < epsilon :
#         average = (upperSum + lowerSum)/2
#         print('I = {} +- {}' .format(average, difference))
#         return True 
#     else :
#         print('Difference > epsilon')
#         print('Upper: ', upperSum, ' Lower: ', lowerSum)
#         print('Try higher n-value')
#         return False
    
# def simpson(a, b, n) : 
#     assert n % 2 == 0, 'n is not even'

#     epsilon = (b - a)/n
#     tvalues = np.linspace(a, b, n + 1)
    
#     #defining the g-vector as an array 
#     lst = [1] 
#     for i in range(1, n) :
#         if i % 2 == 0 :
#             lst.append(2)
#         else:
#             lst.append(4)
#     lst.append(1)

#     #g-vector converted into an array
#     gvec = np.array(lst, dtype = int)

#     #computing f_0, ... , f_n
#     fvec = np.array([integrand(t) for t in tvalues])

#     #scalar product (dot product of the two vectors)
#     integral = (epsilon / 3) * np.dot(gvec, fvec)

#     #error 
#     #error = -epsilon**4/180

#     return integral # , error

# # parameters for use
# a, b = 0, 7
# n = 1000      
# epsilon = 0.1  

# resultRiemann = intergrateRiemann(a, b, n, epsilon)

# if resultRiemann is not None : 
#     print('Riemann sum: ', resultRiemann)


# resultSimpson = (simpson(a, b, n))
# print('Simpson: ', resultSimpson)

# b

# def r(t) :
#     x = 1 - t**2
#     y = t**3 + t
#     return x, y

# def dr(t) : 
#     dx = -2*t
#     dy = 3*t**2 + 1
#     return dx, dy

# def f(x, y) :
#     return x + np.sin(y)

# def integrand(t) :
#     x, y = r(t)  
#     dx, dy = dr(t)
#     return f(x, y) * np.sqrt(dx**2 + dy**2)  #2D case for the arc length          

# def upperRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     upperBound = 0

#     for i in range(1, n) :
#         upperBound += integrand(tvalue[i]) * dt
#     return upperBound

# def lowerRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     lowerBound = 0

#     for i in range(n - 1) :
#         lowerBound += integrand(tvalue[i]) * dt
#     return lowerBound


# def intergrateRiemann(a, b, n, epsilon) :
#     upperSum = upperRiemannSum(a, b, n)
#     lowerSum = lowerRiemannSum(a, b, n)

#     difference = abs(upperSum - lowerSum)

#     if difference < epsilon :
#         average = (upperSum + lowerSum)/2
#         print('I = {} +- {}' .format(average, difference))
#         return True 
#     else :
#         print('Difference > epsilon')
#         print('Upper: ', upperSum, ' Lower: ', lowerSum)
#         print('Try higher n-value')
#         return False
    
# def simpson(a, b, n) : 
#     assert n % 2 == 0, 'n is not even'

#     epsilon = (b - a)/n
#     tvalues = np.linspace(a, b, n + 1)
    
#     #defining the g-vector as an array 
#     lst = [1] 
#     for i in range(1, n) :
#         if i % 2 == 0 :
#             lst.append(2)
#         else:
#             lst.append(4)
#     lst.append(1)

#     #g-vector converted into an array
#     gvec = np.array(lst, dtype = int)

#     #computing f_0, ... , f_n
#     fvec = np.array([integrand(t) for t in tvalues])

#     #scalar product (dot product of the two vectors)
#     integral = (epsilon / 3) * np.dot(gvec, fvec)

#     #error 
#     #error = -epsilon**4/180

#     return integral # , error

# # parameters for use
# a, b = 0, 2
# n = 1000      
# epsilon = 0.1  

# resultRiemann = intergrateRiemann(a, b, n, epsilon)

# if resultRiemann is not None : 
#     print('Riemann sum: ', resultRiemann)


# resultSimpson = (simpson(a, b, n))
# print('Simpson: ', resultSimpson) 

# # c 

# def r(t) :
#     x = np.sin(t)
#     y = np.cos(t)
#     z = np.sin(2*t)*np.cos(3*t)
#     return x, y, z

# def dr(t) : 
#     dx = np.cos(t)
#     dy = -np.sin(t)
#     dz = 2*np.cos(2*t)*np.cos(3*t) - 3*np.sin(2*t)*np.sin(3*t)
#     return dx, dy, dz

# def f(x, y, z) :
#     return x*y + x*z + y*z

# def integrand(t) :
#     x, y, z= r(t)  
#     dx, dy, dz = dr(t)
#     return f(x, y, z) * np.sqrt(dx**2 + dy**2 + dz**2)  #3D case for the arc length          

# def upperRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     upperBound = 0

#     for i in range(1, n) :
#         upperBound += integrand(tvalue[i]) * dt
#     return upperBound

# def lowerRiemannSum(a, b, n) :
#     tvalue = np.linspace(a, b, n)
#     dt = (b - a)/n
#     lowerBound = 0

#     for i in range(n - 1) :
#         lowerBound += integrand(tvalue[i]) * dt
#     return lowerBound


# def intergrateRiemann(a, b, n, epsilon) :
#     upperSum = upperRiemannSum(a, b, n)
#     lowerSum = lowerRiemannSum(a, b, n)

#     difference = abs(upperSum - lowerSum)

#     if difference < epsilon :
#         average = (upperSum + lowerSum)/2
#         print('I = {} +- {}' .format(average, difference))
#         return True 
#     else :
#         print('Difference > epsilon')
#         print('Upper: ', upperSum, ' Lower: ', lowerSum)
#         print('Try higher n-value')
#         return False
    
# def simpson(a, b, n) : 
#     assert n % 2 == 0, 'n is not even'

#     epsilon = (b - a)/n
#     tvalues = np.linspace(a, b, n + 1)
    
#     #defining the g-vector as an array 
#     lst = [1] 
#     for i in range(1, n) :
#         if i % 2 == 0 :
#             lst.append(2)
#         else:
#             lst.append(4)
#     lst.append(1)

#     #g-vector converted into an array
#     gvec = np.array(lst, dtype = int)

#     #computing f_0, ... , f_n
#     fvec = np.array([integrand(t) for t in tvalues])

#     #scalar product (dot product of the two vectors)
#     integral = (epsilon / 3) * np.dot(gvec, fvec)

#     #error 
#     #error = -epsilon**4/180

#     return integral # , error

# # parameters for use
# a, b = 0, 2
# n = 1000      
# epsilon = 0.1  

# resultRiemann = intergrateRiemann(a, b, n, epsilon)

# if resultRiemann is not None : 
#     print('Riemann sum: ', resultRiemann)


# resultSimpson = (simpson(a, b, n))
# print('Simpson: ', resultSimpson) 

# # Exercise 16 Partial functions 

# from sympy import symbols, integrate, diff

# x, y = symbols('x y') # creating symbolic variables x y

# def F(x, y) :
#     P = 3*x**2 - 2*x*y - y**2
#     Q = 3*y**2 - x**2 - 2*x*y**2
#     return P, Q

# #Exercise 19 Double integrals 

# def f_a(x, y) :
#     return x*(1 + x)*y + y**3

# def f_b(x, y) :
#     return np.sin(x)**5 - x*y*np.cos(y**3)

# def f_c(x, y) : 
#     return np.exp(-y*np.cos(x)**2) * np.sin(y)

# def f_d(x, y) :
#     return np.log(x + y) / (np.sqrt(x*y) * np.exp(-np.cos(x)))

# def simpson2D(f, xx, yy, n) : 
#     a, b = xx               # xx denotes the x - intervals
#     c, d = yy               # yy denotes the y - intervals
#     DeltaX = (b - a) / n
#     DeltaY = (d - c) / n

#     x = np.linspace(a, b, n + 1)
#     y = np.linspace(c, d, n + 1)
#     x, y = np.meshgrid(x, y)

#     F = f(x, y)

#     # the g-vector 
#     lst = [1]
#     for i in range(1, n) :
#         if i % 2 == 0 :
#             lst.append(2)
#         else:
#             lst.append(4)
#     lst.append(1)

#     gvec = np.array(lst, dtype=int)
    
#     Int = (DeltaX*DeltaY/9)*(gvec.dot(F)).dot(gvec.T)

#     return Int # Int is the integral

# #a
# xx_a = [-2, 4]
# yy_a = [0, np.pi]
# results_a = simpson2D(f_a, xx_a, yy_a, 10)
# print("Intergral part a): ", results_a)

# #b
# xx_b = [0, 2*np.pi]
# yy_b = [0, 2*np.pi]
# results_b = simpson2D(f_b, xx_b, yy_b, 10)
# print("Intergral part b): ", results_b)

# #c
# xx_c = [-np.pi, 3*np.pi]
# yy_c = [0, 1/np.pi]
# results_c = simpson2D(f_c, xx_c, yy_c, 10)
# print("Intergral part c): ", results_c)

# #d
# xx_d = [1, (np.pi**3)/5]
# yy_d = [1, 29*np.pi]
# results_d = simpson2D(f_d, xx_d, yy_d, 10)
# print("Intergral part d): ", results_d)


#Exercise 20

# using the code from double integral chapter and use that for polar coordinates

# f(rho, theta)
def f_polar(rho, theta) : 
    return rho              # becaouse we only use rho in the polar integral (note this is overleaf)

def simpson2D_polar(f, r_bound, theta_bound, n) : 
    r_min , r_max = r_bound                         # radius bound a < r < b
    theta_min, theta_max = theta_bound              # theta bound  ex. 0 < theta < 2pi

    dr = (r_max - r_min) / n                        # step size for the radius 
    dtheta = (theta_max - theta_min) / n            # step size for theta 


    # making grid point for both rho and theta 
    rvalues = np.linspace(r_min, r_max, n + 1)
    thetavalues = np.linspace(theta_min, theta_max, n + 1)

    r, theta = np.meshgrid(rvalues, thetavalues)

    F = f(r, theta)

    lst = [1]
    for i in range (1, n) :
        if i % 2 == 0 :
            lst.append(2)
        else:
            lst.append(4)
    lst.append(1)

    gvec = np.array(lst, dtype=int)

    Int = (dr * dtheta / 9) * (gvec.dot(F * r)).dot(gvec.T) # we need to multiply the F with the Jacobian factor (rho)

    return Int 

# a

R = np.exp(np.sqrt(np.log(7) + 39*np.pi))
results_a = simpson2D_polar(f_polar, [0, R], [0, 2*np.pi], 10) # here we use the whole circle 
print('The area of disk in part a is: ', results_a)

# b 

t_min = np.pi/7 
t_max = (13*np.pi)/19 
# R is the same as in part a) but we are finding a sector this time 
results_b = simpson2D_polar(f_polar, [0, R], [t_min, t_max], 10)
print('The area of disk in part b is: ', results_b)

# c

# here we have two radiuses, and inner one and an outer one

r_inner = np.sqrt(np.log(2 + np.pi)**2) # x**2 + y**2 < r**2, this is why the radius is in a square root 
r_outer = R                             # the r is the same as in a so i use it here too
results_c = simpson2D_polar(f_polar, [r_inner, r_outer], [0, 2*np.pi], 10) # again we use the whole circle 
print('The area of disk in part c is: ', results_c)










