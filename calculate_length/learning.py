# import numpy as np
#
# A =  np.array([[56.0,0.0,4.4,68.0],
#                [1.2,104.0,52.0,8.0],
#                [1.8,135.0,99.0,0.9]
# ])
#
# cal = A.sum(axis=0)
# # print(cal)
#
# percentage = 100*A/(cal.reshape(1,4))
# print(percentage)
import asyncio.trsock

import numpy as np

x = np.array([5,6,7,8,9],
             [5,6,7,8,9],)

y = np.array([5,6,7,8,9],
             [5, 6, 7, 8, 9], )

w = 0
b = 0
# w =
for i in range(5):

    z = np.dot((np.transpose(w)), x) + b
    # print(z)
    sigmiod = 1/(1+(np.exp(-z)))

    A = sigmiod
    dz = A-y


    # print(dz)

    dw = 1/5*(x*(np.transpose(dz)))
    db = 1/5*(np.sum(dz))

    learning_rate = 0.0000000001
    w+= w-(learning_rate*dw)
    b+= b-(learning_rate*db)
    dz = A-y


    print(dz)
