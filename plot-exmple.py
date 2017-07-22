import numpy as np
'''
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2 * np.pi * t)
plt.plot(t, s1)

cursor = Cursor(plt.gca(), horizOn=True, color='r', lw=1)
plt.show()

import numpy as np

import matplotlib

matplotlib.use('Agg')

from matplotlib.pyplot import plot,show, savefig

x = np.linspace(-4, 4, 30)
y = np.sin(x);

x = plot(x, y, '--*b')
#show()
savefig('d:\MyFig.png')
'''
import numpy as np
from matplotlib import pyplot as plt


random_image = np.random.random([500, 500])
print(random_image)
plt.ion()
plt.ioff()
print "------------------------1"

plt.imshow(random_image, cmap='gray')
plt.colorbar()
plt.draw()
print "------------------------2"
t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2 * np.pi * t)
plt.plot(t, s1)

plt.draw()
