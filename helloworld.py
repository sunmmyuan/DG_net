import numpy as np
import matplotlib.pyplot as plt
print('helloworld')


x = np.linspace(0, 2*np.pi, 1000)
y = np.sin(x)

plt.plot(x, y)
plt.show()
