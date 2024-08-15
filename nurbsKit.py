import math
import numpy as np
from support import bernstein, basis, visualization, common

class bezier(bernstein, common):
    '''bezier class, including the rational bezier feature'''
    def __init__(self, ctrlpts):
        super().__init__()
        self.ctrlpts = np.array(ctrlpts) # the control points
        [rown, coln] = np.shape(ctrlpts)
        self.p = rown - 1 # bezier curve's degree
        self._weights = np.ones(rown) #default it's all one
        self.dimension = coln # It is either 3 or 2.
        self.sampleSize = 50 # default number of interpolation steps
        self.__update_ctrlptsW()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = np.array(value)
        self.__update_ctrlptsW()

    def __update_ctrlptsW(self):
        rown = self.p + 1
        self.ctrlptsW = np.column_stack((self.ctrlpts * self.weights.reshape(rown, 1), self.weights))

    def __pStarDer(self, u, order):
        pStarDer = np.zeros(self.dimension)
        for i in range(self.p+1):
            pStarDer += self.weights[i] * bernstein.derivative(i, self.p, u, order) * self.ctrlpts[i,:]

        return pStarDer

    def __wDer(self, u, order):
        wDer = 0.0
        for i in range(self.p+1):
            wDer += self.weights[i] * bernstein.derivative(i, self.p, u, order)

        return wDer

    def evaluate(self, u):
        '''evaluate a bezier curve a single u'''
        curvePt = np.zeros(self.dimension + 1)
        for i in range(self.p + 1):
            curvePt += self.ctrlptsW[i,:] * bernstein.bernsteinPoly(i, self.p, u)

        pt = curvePt[0:-1] / curvePt[-1]

        return pt

    def derivative(self, u, order):
        '''calculate the order-th derivative of a bezier curve at designated u'''
        if order < 0:
            raise ValueError('derivative order must be >= 0')

        n = order + 1
        m = n if order <= self.p else (self.p + 1)
        wDers = np.zeros([m, 1])
        bezierDers = np.zeros([n, self.dimension])
        for i in range(m):
            wDers[i] = self.__wDer(u, i)
            bezierDers[i, :] = self.__pStarDer(u, i)

        for k in range(n):
            for i in range(k):
                j = k - i
                if j <= self.p:
                    bezierDers[k, :] -= math.comb(k, j) * bezierDers[i,:] * wDers[j]

            bezierDers[k, :] /= wDers[0]

        return bezierDers

    def trace(self):
        '''calculate the trace of a bezier curve'''
        return common.trace(self.evaluate, self.sampleSize, self.dimension)

    def length(self, a = 0, b = 1):
        '''calculate the arc length of the bezier curve,
        using Legendre-Gauss quadrature'''
        return common.arcLen(self.derivative, a, b)

    def curvature(self, u):
        '''calculate the curvature at u'''
        return common.curvature(self.derivative, u)

    # plot a bezier
    def vis(self):
        return common.vis(self.trace(), self.dimension, self.ctrlpts, self.p)


class nurbs(basis, common):
    def __init__(self, ctrlpts, knots, weights, degree):
        super().__init__()
        self.ctrlpts = np.array(ctrlpts)
        self.U = np.array(knots)
        self.p = degree
        rown, coln = np.shape(ctrlpts)
        self.n = rown - 1
        self._weights = np.array(weights)
        self.dimension = coln
        self.sampleSize = 50  # default number of interpolation steps
        self.__update_ctrlptsW()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = np.array(value)
        self.__update_ctrlptsW()

    def __update_ctrlptsW(self):
        rown = self.n + 1
        self.ctrlptsW = np.column_stack((self.ctrlpts * self.weights.reshape(rown, 1), self.weights))

    def __wDers(self, span, basisDers, k):
        wDers = np.zeros(k)
        for i in range(k):
            for j in range(self.p + 1):
                wDers[i] += basisDers[i, j] * self.weights[span - self.p + j]

        return wDers

    def __pStarDers(self, span, basisDers, k):
        pStarDers = np.zeros([k, self.dimension])
        for i in range(k):
            for j in range(self.p + 1):
                pStarDers[i, :] += basisDers[i, j] * self.weights[span - self.p + j] * self.ctrlpts[span - self.p + j, :]

        return pStarDers

    def __calcRationalBSplineAndBezierDers(self, ders, wDers, order):
        for i in range(order + 1):
            for j in range(1, i + 1):
                if (j <= self.p):
                    ders[i, :] -= math.comb(i, j) * ders[i - j, :] * wDers[j]
            ders[i, :] /= wDers[0]

        return ders

    def evaluate(self, u):
        i = basis.findSpan(self.n, self.p, u, self.U)
        N = basis.evaluate(i, self.p, u, self.U)
        tempPt = np.zeros(self.dimension + 1)
        for j in range(self.p+1):
            tempPt += N[j] * self.ctrlptsW[i-self.p+j, :]

        pt = tempPt[0:-1] / tempPt[-1]

        return pt

    def derivative(self, u, order):
        '''
        @:parameter
        u: the parameter u
        order: the order of the derivative

        @:return: derivatives
        '''
        if order < 0:
            raise ValueError('derivative order must be >= 0')

        k = (order+1) if order <= self.p else (self.p+1)
        nurbsDers = np.zeros([order+1, self.dimension])
        span = basis.findSpan(self.n, self.p, u, self.U)
        basisDers = basis.derivatives(span, self.p, u, self.U, order)
        wDers = self.__wDers(span, basisDers, k)
        nurbsDers[0:k,:] = self.__pStarDers(span, basisDers, k)
        nurbsDers = self.__calcRationalBSplineAndBezierDers(nurbsDers, wDers, order)

        return nurbsDers

    def length(self, a=0, b=1):
        '''calculate the arc length of nurbs curves,
        using Legendre-Gauss quadrature'''
        return common.arcLen(self.derivative, a, b)

    def curvature(self, u):
        return common.curvature(self.derivative, u)

    def trace(self):
        '''calculate the trace of a nurbs curve'''
        return common.trace(self.evaluate, self.sampleSize, self.dimension)

    def vis(self):
        return common.vis(self.trace(), self.dimension, self.ctrlpts, self.p)
