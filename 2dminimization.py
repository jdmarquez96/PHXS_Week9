import numpy as np
from scipy.optimize import minimize as miny
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = 2.5
b = -0.5

def f(x):
        return (x[0] + a)**2 + (x[1] + b)**2


guess = [3, 3]

minimize = miny(f, x0 = guess)



print(minimize)


def g(x, y):
	return (x + a)**2 + (y - b)**2



x = np.arange(-3, 3, 0.01)
y = np.arange(-3, 3, 0.01)

xg, yg = np.meshgrid(x, y)

plt.figure()
plt.imshow(g(xg, yg), extent = [-3, 3, -3, 3], origin = "lower", label="g(x,y)")
plt.scatter(minimize.x[0], minimize.x[1], label="minimum point x = %.2f" %(minimize.x[0])     )
plt.scatter(minimize.x[0], minimize.x[1], label="minimum point y = %.2f" %(minimize.x[1])     )
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xg, yg, g(xg, yg), rstride=1, cstride=1,cmap=plt.cm.jet, linewidth=0, antialiased=False,label="g(x,y)")
ax.scatter(minimize.x[0],minimize.x[1], g(minimize.x[0], minimize.x[1]), color = "red",label="minimum point")
#ax.scatter(-2.5,0.5,0, color="red")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('g(x,y)')
plt.savefig("3dplot.png")


#save as a picture because showing it live tanks my processing power (everything become slow and laggy)
