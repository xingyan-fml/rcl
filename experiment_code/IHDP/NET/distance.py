import tensorflow as tf
import numpy as np
SQRT_CONST = 1e-10
def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))
def lindisc(X,p,t):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,axis=0)
    mean_treated = tf.reduce_mean(Xt,axis=0)

    c = tf.square(2*p-1)*0.25
    f = tf.sign(p-0.5)

    mmd = tf.reduce_sum(tf.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc, axis=0)
    mean_treated = tf.reduce_mean(Xt, axis=0)
    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    Kcc = tf.exp(-pdist2sq(Xc,Xc)/tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc,Xt)/tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt,Xt)/tf.square(sig))

    m = tf.cast(tf.shape(Xc)[0], dtype=tf.float32)
    n = tf.cast(tf.shape(Xt)[0], dtype=tf.float32)

    mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(Kcc)-m)
    mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(Kct)
    mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.compat.v1.reduce_sum(tf.square(X),1,keep_dims=True)
    ny = tf.compat.v1.reduce_sum(tf.square(Y),1,keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def pop_dist(X,t):
    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.cast(tf.shape(Xc)[0],dtype=tf.float32)
    nt = tf.cast(tf.shape(Xt)[0],dtype=tf.float32)

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    # M = tf.Print(M, [tf.shape(Xt), tf.shape(Xc), tf.shape(M), tf.shape(M[0:1,:])], "r, c, m, d - shapes")
    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    col = tf.concat([delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))], 0)
    # row = tf.Print(row, [tf.shape(row), tf.shape(col), tf.shape(M), tf.shape(M[0:1,:])], "r, c, m, d - shapes")

    Mt = tf.concat([M,row], 0)
    Mt = tf.concat([Mt,col], 1)

    ''' Compute marginal vectors '''
    a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))], 0)
    b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D, Mlam