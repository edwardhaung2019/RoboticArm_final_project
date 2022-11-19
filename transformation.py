import numpy as np
import math

def trans(angle):

    q = np.zeros(6)

    for i in range(6):
        q[i] = angle[i]

    d1 = 0.1451
    a2 = 0.4290
    a3 = 0.4115
    d4 = -0.1222
    d5 = 0.1060
    d6 = 0.1144

    DEG2RAD = 0.01745329252
    _PI_2 = 0.5 * math.pi


    for i in range(q.size):
        q[i] = q[i] * DEG2RAD

    T = np.zeros((4, 4))

    c1 = math.cos(q[0])
    s1 = math.sin(q[0])
    c2 = math.cos(q[1] - _PI_2)
    s2 = math.sin(q[1] - _PI_2)
    c3 = math.cos(q[2])
    s3 = math.sin(q[2])
    c4 = math.cos(q[3] + _PI_2)
    s4 = math.sin(q[3] + _PI_2)
    c5 = math.cos(q[4])
    s5 = math.sin(q[4])
    c6 = math.cos(q[5])
    s6 = math.sin(q[5])
    cp = math.cos(q[1] + q[2] + q[3])
    sp = math.sin(q[1] + q[2] + q[3])

    T[0, 0] = c1 * sp * s6 - s1 * s5 * c6 + c1 * cp * c5 * c6
    T[0, 1] = c1 * sp * c6 + s1 * s5 * s6 - c1 * cp * c5 * s6
    T[0, 2] = c1 * cp * s5 + s1 * c5
    T[0, 3] = c1 * (a2 * c2 + a3 * c2 * c3 - a3 * s2 * s3) - d4 * s1 + d6 * c5 * s1 \
              + d5 * c1 * (c4 * (c2 * s3 + c3 * s2) + s4 * (c2 * c3 - s2 * s3)) \
              + d6 * c1 * s5 * (c4 * (c2 * c3 - s2 * s3) - s4 * (c2 * s3 + c3 * s2))
    T[1, 0] = s1 * sp * s6 + c1 * s5 * c6 + s1 * cp * c5 * c6
    T[1, 1] = s1 * sp * c6 - c1 * s5 * s6 - s1 * cp * c5 * s6
    T[1, 2] = s1 * cp * s5 - c1 * c5
    T[1, 3] = s1 * (a2 * c2 + a3 * c2 * c3 - a3 * s2 * s3) + d4 * c1 - d6 * c1 * c5 \
              + d5 * s1 * (c4 * (c2 * s3 + c3 * s2) + s4 * (c2 * c3 - s2 * s3)) \
              + d6 * s1 * s5 * (c4 * (c2 * c3 - s2 * s3) - s4 * (c2 * s3 + c3 * s2))
    T[2, 0] = cp * s6 - sp * c5 * c6
    T[2, 1] = cp * c6 + sp * c5 * s6
    T[2, 2] = -sp * s5
    T[2, 3] = d1 - a2 * s2 + d5 * (c4 * (c2 * c3 - s2 * s3) - s4 * (c2 * s3 + c3 * s2)) \
              - a3 * c2 * s3 - a3 * c3 * s2 - d6 * s5 * (c4 * (c2 * s3 + c3 * s2) + s4 * (c2 * c3 - s2 * s3))
    T[3, 0] = 0
    T[3, 1] = 0
    T[3, 2] = 0
    T[3, 3] = 1

    return T




