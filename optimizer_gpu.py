import cupy as np

class SGD:
    def __init__( self, lr = 0.1 ):
        self.lr = lr

    def update( self, params, grads ):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__( self, lr = 0.1, momentum = 0.9 ):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update( self, params, grads ):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like( val )

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__( self, lr = 0.01 ):
        self.lr = lr
        self.h = None

    def update( self, params, grads ):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like( val )

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / ( np.sqrt( self.h[key] + 1e-7 ) )


class RMSprop:
    def __init__( self, lr=0.0001, decay_rate = 0.99 ):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update( self, params, grads ):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[ key ] = np.zeros_like( val )

        for key in params.keys():
            self.h[ key ] *= self.decay_rate
            self.h[ key ] += ( 1 - self.decay_rate ) * grads[ key ] * grads[ key ]
            params[ key ] -= self.lr * grads[ key ] / ( np.sqrt( self.h[ key ] ) + 1e-7 )


class AdaDelta:
    def __init__( self, decay_rate = 0.99 ):
        self.decay_rate = decay_rate
        self.v = None
        self.s = None

    def update( self, params, grads ):
        if self.v is None:
            self.v = {}
            self.s = {}
            for key,val in params.items():
                self.v[ key ] = np.zeros_like( val )
                self.s[ key ] = np.zeros_like( val )

        for key in params.keys():
            self.v[ key ] *= self.decay_rate
            self.s[ key ] *= self.decay_rate
            self.v[ key ] += ( 1 - self.decay_rate ) * grads[ key ] * grads[ key ]
            self.s[ key ] += ( 1 - self.decay_rate ) * params[ key ] * params[ key ]
            params[ key ] -= grads[key ] * ( np.sqrt( self.s[ key ] ) + 1e-7 ) / ( np.sqrt( self.v[ key ] ) + 1e-7 )

class Adam:
    def __init__( self, lr ):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.iter = 0
        self.m = None
        self.v = None

    def update( self, params, grads, idx ):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[ key ] = np.zeros_like( val )
                self.v[ key ] = np.zeros_like( val )

        self.iter = idx+1
        lr_t = self.lr * np.sqrt( 1.0 - self.beta2 ** self.iter ) / ( 1.0 - self.beta1 ** self.iter )

        for key in params.keys():
            #print(key)
            self.m[ key ] += ( 1 - self.beta1 ) * ( grads[ key ] - self.m[ key ] )
            self.v[ key ] += ( 1 - self.beta2 ) * ( grads[ key ] ** 2 - self.v[ key ] )
            params[ key ] -= lr_t * self.m[ key ] / ( np.sqrt( self.v[ key ] + 1e-7 ) )


class AdaBound:
    def __init__(self, lr, final_lr):
        self.lr = lr
        self.final_lr = final_lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.iter = 0
        self.m = None
        self.v = None
        self.eps = 1e-7

    def update(self, params, grads, epoch):
        if self.m is None and self.v is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter = epoch
        for key in params.keys():
            self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*grads[key]**2
            lower_lr = self.final_lr*(1.0 - 1.0/((1-self.beta2)*self.iter + 1 + self.eps))
            higher_lr = self.final_lr*(1.0 + 1.0/((1-self.beta2)*self.iter + self.eps))
            lr_t = np.clip(self.lr / np.sqrt(self.v[key] + self.eps), lower_lr, higher_lr)
            params[key] -= lr_t*self.m[key]


class Nesterov:
    def __init__( self, lr=0.0001, momentum=0.9 ):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update( self, params, grads ):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[ key ] = np.zeros_like( val )

        for key in params.keys():
            self.v[ key ] *= self.momentum
            self.v[ key ] -= self.lr * grads[ key ]
            params[ key ] += self.momentum * self.momentum * self.v[ key ]
            params[ key ] -= ( 1 + self.momentum ) * self.lr * grads[ key ]


def set_optimizer(opt, lr):
    if opt == "SGD":
        return SGD(lr = lr)
    if opt == "Momentum":
        return Momentum(lr = lr, momentum = 1.0-lr)
    if opt == "Adam":
        return Adam(lr = lr)
    if opt == "AdaBound":
        return AdaBound(lr = 1.0, final_lr = lr)
    else:
        return None
