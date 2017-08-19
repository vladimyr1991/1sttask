
# coding: utf-8

# In[135]:

from math import (sin, exp)
import numpy as np
import scipy as sp
from matplotlib import pylab as plt

x = np.arange(1.,16.)

def f_x (x):
    return sin(x/5) * exp(x/10) + 5 * exp(-x/2)
func=[]
for i in x:
    f_x(i)
    func.append(f_x(i))
print (func)

plt.plot (x,func)
plt.show()


# In[152]:

a = np.array([[1, 1], [1, 15]])
b = np.array([[f_x(1)], [f_x(15)]])
print(a.shape)
print(b.shape)

w = np.linalg.solve(a,b)
w_0 = round(w[0], 2)
w_1 = round(w[1], 2)
print w_0, w_1


# ### Построим аппроксимированную функцию 2 степени

# In[137]:

def k_2(x):
    return round(w_0 + w_1*x, 2)

func_k_2=[]

for i in x:
    k_2(i)
    func_k_2.append(k_2(i))
print (func_k_2)

plt.plot (x,func_k_2)
plt.show()


# # Добавим 3 уровнение в систему (8)

# In[161]:

a = np.array([[1, 1], [1, 8], [1, 15]],dtype = float)
b = np.array([[f_x(1)], [f_x(8)], [f_x(15)]])
print(a.shape)
print(b.shape)

w = np.linalg.solve(a, b)
w_0 = round(w[0], 2)
w_1 = round(w[1], 2)
print w_0, w_1


# In[ ]:



