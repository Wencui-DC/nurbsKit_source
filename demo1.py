# demonstration for NURBS
import nurbsKit

ctrlpts = [0, 2, 15], [2, 2, 10], [2, 0, 5], [0, 0, 0], [9, 9, 9]
knots = [0, 0, 0, 0, .3, 1, 1, 1, 1]
weights = [1, 1, 1, 1, 1]
degree = 3
u = 0.76

# create a nurbs
nurbs = nurbsKit.nurbs(ctrlpts, knots, weights, degree)
# adjust the weights after the nurbs initiated
nurbs.weights = [2, 1, 1, 1, 3]


# evaluate the nurbs at u
pt = nurbs.evaluate(u)
print('C(%.2f) = [%.4f, %.4f, %.4f]' % (u, pt[0], pt[1], pt[2]))

# calculate its derivatives, in any order
order = 2
nurbsDers = nurbs.derivative(u, order)
for i in range(order+1):
    print("C(%.2f)的%d阶导数 = [%.4f, %.4f, %.4f]" % (u, i, nurbsDers[i][0], nurbsDers[i][1], nurbsDers[i][2]))

# calculate its arc length, the default u interval is [0,1].
# You can change it to any interval, ex: nurbs.length(0,0.33) or nurbs.length(0.15, 0.79)
len = nurbs.length()
print('nurbs length is %.4f' % len)

# calculate the curvature at u
k = nurbs.curvature(u)
print('Curvature at %.2f is %.4f' % (u,k))

# plot the nurbs
nurbs.vis()


