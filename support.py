import math
import numpy as np
import matplotlib.pyplot as plt

# global variables for Legendre-Gauss quadrature
abscissae = [0,	-0.201194093997435,	0.201194093997435,
             -0.394151347077563, 0.394151347077563,	-0.570972172608539,
             0.570972172608539,	-0.724417731360170,	0.724417731360170,
             -0.848206583410427, 0.848206583410427, -0.937273392400706,
             0.937273392400706,-0.987992518020485, 0.987992518020485]

weights_LG = [0.202578241925561, 0.198431485327112, 0.198431485327112,
           0.186161000015562, 0.186161000015562, 0.166269205816994,
           0.166269205816994, 0.139570677926154, 0.139570677926154,
           0.107159220467172, 0.107159220467172, 0.0703660474881081,
           0.0703660474881081, 0.0307532419961173, 0.0307532419961173]

class bernstein:
    def __init__(self):
        super().__init__()

    @staticmethod
    def bernsteinPoly(i, p, u):
        '''calculate the bernstein polynomial

        @:parameter
        i: the index of u_i for u
        p: the curve degree
        u: the parameter u

        @:return
        bernstein function's value
        '''
        bernsteinFun = 0.
        if 0 <= i <= p:
            bernsteinFun = math.comb(p, i) * np.power(u, i) * np.power(1 - u, p - i)

        return bernsteinFun

    @staticmethod
    def derivative(i, p, u, order):
        '''calculate the derivative of bernstein polynomial'''
        derB = 0.0
        newP = p - order
        start = max(0, i-newP)
        end = min(i, order) + 1
        for j in range(start, end):
            newI = i - j
            derB += np.power(-1, order+j) * math.comb(order, j) * bernstein.bernsteinPoly(newI, newP, u)

        if derB != 0 and order > 0:
            derB *= math.perm(p, order)

        return derB

class basis:
    @staticmethod
    def findSpan(n, p, u, U):
        '''find the index of u_i for u
        @:parameter
        n: the number of ctrlpts
        p: the curve degree
        u: the parameter u
        U: the knots vector

        @:return
        index: the index of u_i for u
        '''
        if u == U[n + 1]:
            return n

        low = p
        high = n + 1
        index = (low + high) // 2
        while u < U[index] or u >= U[index + 1]:
            if u < U[index]:
                high = index
            else:
                low = index

            index = (low + high) // 2

        return index

    @staticmethod
    def evaluate(i, p, u, U):
        '''evaluate basis function at u
        @:parameter
        i: the index of u_i for u
        p: the curve degree
        u: the parameter u
        U: the knots vector
        @:return
        N: basis function's value at u
        '''
        N = np.ones([p+1, 1])
        left = np.zeros([p, 1])
        right = np.zeros([p, 1])
        for j in range(1, p+1):
            left[j-1] = u - U[i+1-j]
            right[j-1] = U[i+j] - u
            saved = 0.0;
            for r in range(j):
                temp = N[r] / (right[r] + left[j-r-1])
                N[r] = saved + right[r] * temp
                saved = left[j-r-1] * temp

            N[j] = saved

        return N

    @staticmethod
    def derivatives(i, p, u, U, order):
        m = p + 1
        n = (order+1) if order <= p else m
        ndu = np.zeros([m, m])
        ndu[0,0] = 1.0
        a = np.zeros([2, n])
        left = np.zeros([m,1])
        right = np.zeros([m,1])
        basisDers = np.zeros([n, m])
        for j in range(1, m):
            left[j] = u - U[i+1-j]
            right[j] = U[i+j] - u
            saved = 0.0
            for r in range(j):
                ndu[j, r] = right[r+1] + left[j-r]
                temp = ndu[r, j-1] / ndu[j, r]
                ndu[r, j] = saved + right[r+1]*temp
                saved = left[j-r]*temp

            ndu[j,j] = saved

        for j in range(m):
            basisDers[0,j] = ndu[j,p]

        #this section computes the derivatives
        for r in range(m):
            s1 = 0
            s2 = 1
            a[0,0] = 1.0
            for k in range(1, n):
                d = 0.0
                rk = r-k
                pk = p-k
                if (r >= k):
                    a[s2,0] = a[s1,0] / ndu[pk+1, rk]
                    d = a[s2,0] * ndu[rk, pk]

                j1 = 1 if rk >= -1 else -rk
                j2 = k-1 if r-1 <= pk else p-r
                for j in range(j1, j2+1):
                    a[s2,j] = (a[s1,j] - a[s1, j-1]) / ndu[pk+1, rk+j]
                    d += a[s2,j] * ndu[rk+j, pk]

                if (r <= pk):
                    a[s2,k] = -a[s1,k-1] / ndu[pk+1, r]
                    d += a[s2,k] * ndu[r,pk]

                basisDers[k,r] = d
                j = s1
                s1 = s2
                s2 = j

        r = p
        for k in range(1, n):
            for j in range(m):
                basisDers[k, j] *= r
            r *= (p-k)

        return basisDers

class visualization:
    def __init__(self):
        super().__init__()

    @staticmethod
    def plot2d(trace, ctrlpts, p):
        plt.figure()
        x = np.array(ctrlpts[:, 0])
        y = np.array(ctrlpts[:, 1])
        plt.plot(trace[:, 0], trace[:, 1], color='blue')
        plt.scatter(x, y, s=50, marker='o', color='black')
        plt.plot(x, y, linestyle='dashed', color='black', linewidth=1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.title(f'a bezier curve with degree of {p}')
        plt.show()

        return 0

    @staticmethod
    def plot3d(trace, ctrlpts, p):
        fig = plt.figure()
        x = np.array(ctrlpts[:, 0])
        y = np.array(ctrlpts[:, 1])
        z = np.array(ctrlpts[:, 2])
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trace[:, 0], trace[:, 1], trace[:, 2], color='blue')
        ax.scatter(x, y, z, s=50, marker='o', color='black')
        ax.plot(x, y, z, linestyle='dashed', color='black', linewidth=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1.0, 1.0, 1.0])
        plt.title(f'a bezier curve with degree of {p}')
        plt.show()

        return 0

class common():
    def __init__(self):
        super().__init__()

    @staticmethod
    def arcLen(derivativeFun, a, b):
        '''calculate the arc length of the bezier/nurbs curve,
        using Legendre-Gauss quadrature

        @:parameter
        a: the lower bound of parameter u \in [0,1]
        b: the higher bound of parameter u \in [0,1]

        @:return
        arcLen: the arc length
        '''
        if a < 0 or b > 1:
            raise ValueError('Interval of U is not within [0, 1]')

        coef_1 = (b - a) / 2
        coef_2 = (b + a) / 2
        arcLen = 0
        abscissaeLen = 15
        for i in range(abscissaeLen):
            u = coef_1 * abscissae[i] + coef_2
            ders = derivativeFun(u, 1)
            firstDer = ders[1, :]
            normSquare = np.sum(firstDer ** 2)
            arcLen += weights_LG[i] * math.sqrt(normSquare)

        arcLen *= coef_1

        return arcLen

    @staticmethod
    def curvature(derivativeFun, u):
        ders = derivativeFun(u, 2)
        firstDer = ders[1, :]
        secondDer = ders[2, :]
        k = np.linalg.norm(np.cross(firstDer, secondDer)) / np.linalg.norm(firstDer)**3

        return k

    @staticmethod
    def trace(evaluateFun, sampleSize, dimension):
        '''calculate the trace of a bezier/nurbs curve

        @:return
        trace: the trace (interpolated points) of the curve
        '''
        step = 1 / (sampleSize - 1)
        U = np.arange(0, 1 + step, step)
        trace = np.zeros((sampleSize, dimension))
        for i in range(sampleSize):
            trace[i, :] = evaluateFun(U[i])

        return trace

    @staticmethod
    def vis(trace, dimension, ctrlpts, degree):
        if dimension == 2:
            visualization.plot2d(trace, ctrlpts, degree)
        elif dimension == 3:
            visualization.plot3d(trace, ctrlpts, degree)
        else:
            raise ValueError('The curve dimension is neither 2 nor 3!')

        return 0