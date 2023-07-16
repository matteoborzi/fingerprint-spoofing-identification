import numpy


def dot(x1, x2, K):
    return numpy.dot(x1.T,x2)


def poly(c, d):
    def p(x1, x2, K):
        return (numpy.dot(x1.T, x2) + c) ** d + K**2
    return p


def rbf(gamma):
    def g(x1, x2, K):
        kern = numpy.zeros((x1.shape[1], x2.shape[1]))
        for i in range(x1.shape[1]):
            for j in range(x2.shape[1]):
                kern[i,j] = numpy.exp(-gamma * (numpy.linalg.norm(x1[:, i] - x2[:, j]) ** 2))
        return kern + K**2
    return g