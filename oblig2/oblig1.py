import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import argparse
from textwrap import wrap

def data1(plot = True):
    n = 30
    start = -2
    stop = 2
    x = np.linspace(start,stop,n)
    eps = 1
    np.random.random(1)
    r = np.random.random(n) * eps
    y = x*(np.cos(r+0.5*x**3)+np.sin(0.5*x**3))
    if plot:
        plt.plot(x,y,'o')
        plt.show()
    return x,y

def data2(plot = True):
    n = 30
    start = -2
    stop = 2
    x = np.linspace(start,stop,n)
    eps = 1
    # rng(1)
    r = np.random.random(n) * eps
    y = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r
    if plot:
        plt.plot(x,y,'o')
        plt.show()
    return x,y

def QR_solve(A,b):
    """Solves Ax = b with QR-factorization"""

    # with mode = 'economic', R==R1 
    Q,R = linalg.qr(A, mode = 'economic')
    M = R.shape[1]
    R1 = R
    c = Q.T @ b
    c1, c2 = c[:M], c[M:]

    x = backward(R1,c1)
    return x

def backward(U,b):
    M = U.shape[0]
    x = np.zeros(M)
    x[-1] = b[-1] / U[-1,-1] 
    for i in range(2,M+1):
        x[-i] = (b[-i] - np.sum(U[-i,-i+1:]*x[-i+1:])) / U[-i,-i] 
    return x

def forward(L,b):
    M = L.shape[0]
    x = np.zeros(M)

    x[0] = b[0]/L[0,0]
    for i in range(1,M):
        x[i] = (b[i] - np.sum(L[i,:i-1]*x[:i-1])) / L[i,i] 
    return x


def ex1(args):
    print('ex1')
    x,y = data1(plot=False)
    deg = 3
    X = np.array([x**i for i in range(deg+1)]).T
    
    beta = QR_solve(X,y)
    plt.plot(x,y,'o', label='data')
    plt.plot(x,X@beta,'-', label = 'order {}'.format(deg))
    plt.legend()
    plt.show()

def ex2(args):
    print('ex2')
    x,y = data1(plot=False)
    deg = 5
    X = np.array([x**i for i in range(deg+1)]).T
    B = X.T @ X
    beta = cholesky_solve(X, y)
    plt.plot(x,y,'o', label='data')
    plt.plot(x,X@beta,'-', label = 'order {}'.format(deg))
    plt.legend()
    plt.show()

def cholesky_solve(A,b):
    """Solves Ax=b through normal equations A.T Ax = A.T b, using cholesky
    factorization, solving R y = A.T b with forward sub and then R.T x = y """
    # Solve 
    # Solve R^T x = y
    # remember, R is lower diag
    R = cholesky(A.T@A)
    x = backward(R.T, forward(R, A.T @ b))
    return x



def cholesky(A, RR = True):
    Ak = A.copy()
    n = A.shape[0]
    L = np.zeros(Ak.shape)
    D = np.zeros(Ak.shape[0])
    for k in range(n):
        L[:,k] = Ak[:,k]
        D[k]   = Ak[k,k]
        L[:,k] = L[:,k]/D[k]
        Ak -= D[k]*np.outer(L[:,k], L[:,k])
    if RR:
        R = L  * np.sqrt(D)
        return R
    else: 
        return L, np.diag(D)

def K2_condition(A):
    '''Calculates the K-2 condition of solving a linear system Ax = b using
    the singular values of A'''

    svd = linalg.svd(A)
    sing_vals = svd[1]
    return (np.max(sing_vals) / np.min(sing_vals))

def ex3(args):
    x,y = data1(plot=False)
    deg = 5
    X = np.array([x**i for i in range(deg+1)]).T
    
    Q,R = linalg.qr(X, mode = 'economic')
    #svd = linalg.svd()
    Rchol = cholesky(X.T@X)
    print(K2_condition(Rchol))
    print(K2_condition(Rchol.T))
    print(K2_condition(Rchol.T)**2)
    print(K2_condition(Rchol @ Rchol.T))

    print(K2_condition(X))

    print(R.shape)
    # K2_condition(R)
def produce_results(args):
    for i,datafunc in enumerate([data1, data2]):
        fig, axes = plt.subplots(1,2, figsize = [9,3])

        x,y = datafunc(plot=False)
        for deg, ax in zip([3,8], axes):
            print(deg)
            X = np.array([x**i for i in range(deg+1)]).T
            
            # QR
            beta = QR_solve(X,y)
            ax.plot(x,y,'o', )
            ax.plot(x,X@beta,'-', label='QR solver')

            # Cholesky
            B = X.T @ X
            beta = cholesky_solve(X, y)
            ax.plot(x,X@beta,'--', label='Cholesky solver')
            ax.legend()
            ax.set_title('Degree = {}'.format(deg))
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y = f(x)$')
        # fig.suptitle(i+1)
        fig.tight_layout()
        print(i+1)
        plt.savefig(f'data{i+1}.pdf')
        plt.show()


def main(args):
    """Either runs all parts or just one"""
    parts = {1:ex1, 2: ex2, 3:ex3, 4:produce_results}
    if args.part == 0:
        ex1(args)
        ex2(args)
        ex3(args)
        produce_results(args)
    else:
        parts[args.part](args)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p','--part', type=int,default=0, 
            choices = [0,1,2,3,4])
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    main(args)
