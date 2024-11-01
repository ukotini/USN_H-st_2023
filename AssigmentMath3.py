import numpy as np
import matplotlib.pyplot as plot 

#exercise  1

#a)

fig = plot.figure(figsize =(10, 10))
ax = fig.add_subplot()


#define the parameters 
t = np.linspace(-2, 2, 100)
x = t**5
y = 4 - t**3

ax.plot(x, y, label='A parametric curve')
ax.legend()
plot.show()
