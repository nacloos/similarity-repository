import torch as ch


def centering(K):
    n = K.shape[0]
    unit = ch.ones([n, n], device=K.device).float()
    I = ch.eye(n, device=K.device).float()
    H = I - unit / n

    unit, I = None, None
    ch.cuda.empty_cache()

    return ch.matmul(ch.matmul(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = ch.matmul(X, X.T)
    KX = ch.diag(GX) - GX + (ch.diag(GX) - GX).T
    if sigma is None:
        mdist = ch.median(KX[KX != 0])
        sigma = ch.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = ch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return ch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    return ch.sum(centering(ch.matmul(X, X.T)) * centering(ch.matmul(Y, Y.T)))


def linear_CKA(X, Y):
    return linear_HSIC(X.float(), Y.float()) / \
        (ch.sqrt(linear_HSIC(X.float(), X.float())) * \
            ch.sqrt(linear_HSIC(Y.float(), Y.float())))

def kernel_CKA(X, Y, sigma=None):
    return kernel_HSIC(X.float(), Y.float(), sigma) / \
        (ch.sqrt(kernel_HSIC(X.float(), X.float(), sigma)) * \
            ch.sqrt(kernel_HSIC(Y.float(), Y.float(), sigma)))

class WrapperCKA:
    
    def __init__(self, cka_func) -> None:
        self.cka_func = cka_func
        self.val = None

    def reset(self) -> None:
        self.val = None
        ch.cuda.empty_cache()

    def __call__(self, X: ch.Tensor, Y: ch.Tensor) -> None:
        self.val = self.cka_func(X, Y).detach().cpu().item()

    def value(self, reset: bool = False) -> ch.Tensor:
        v = self.val
        if reset:
            self.reset()
        return v


if __name__=='__main__':
    X = ch.random.randn(100, 64)
    Y = ch.random.randn(100, 64)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))