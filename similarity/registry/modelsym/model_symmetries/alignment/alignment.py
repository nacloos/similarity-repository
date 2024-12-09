import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy
from typing import  Tuple, List
from scipy import optimize as spo

class Features():
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, m, x, y):
        m.features = y
        self.features = y

    def close(self):
        self.hook.remove()


# def feature_collector(dataloader: DataLoader,
#     model1: nn.Module, model2: nn.Module,
#     layer_type = None,
#     device: torch.device = torch.cuda.current_device(),
#     progress: bool = False,
#     debug = False,
#     batched=False):
#     model1.eval()
#     model1.to(device)
#     model2.eval()
#     model2.to(device)
#     h1 = list(model1.modules())
#     h2 = list(model2.modules())
#     if layer_type != None:
#         h1 = [x for x in h1 if isinstance(x, layer_type)]
#         h2 = [x for x in h2 if isinstance(x, layer_type)]
#     print('module types:', [type(x) for x in h1], [type(x) for x in h2])
#     H1, H2 = [[Features(x) for x in h] for h in [h1, h2]]
#     features1, features2 = [[[] for x in H] for H in [H1, H2]]
#     loop = enumerate(dataloader)
#     if progress:
#         loop = tqdm(loop, total=len(dataloader), desc='collecting features')
#     with torch.no_grad():
#         for i, (x, y) in loop:
#             if debug and i >= 1:
#                 break
#             x = x.to(device)
#             y1, y2 = model1(x), model2(x)

#             for l, h in zip(features1, H1):
#                 l.append(h.features.cpu())
#             for l, h in zip(features2, H2):
#                 l.append(h.features.cpu())

#     if batched:
#         return [features1, features2]
#     else:
#         return [[torch.cat(x, dim=0) for x in f] for f in [features1, features2]]

def wreath_procrustes(features1: List[torch.Tensor],
    features2: List[torch.Tensor], diagonly=True,
    dims: Tuple[int] = (1, 1), device=torch.device('cpu'), progress=True, batched=False):
    # TODO: similar thing for CKA via quadratic assignment
    print("calculating wreath procrustes")
    def pairwise_procrustes(H1: torch.Tensor, H2: torch.Tensor):
        with torch.no_grad():
            H1, H2 = deepcopy(H1), deepcopy(H2)
            H1, H2 = H1.to(device), H2.to(device)
            H1 = H1.transpose(1, dims[0])
            H2 = H2.transpose(1, dims[1])
            d1, d2 = [[i for i in range(len(h.shape)) if i != 1]
                for  h in (H1, H2)]
            if H1.shape[-1] != H2.shape[-1]:
                H1, H2 = sorted([H1, H2], key = lambda x: x.shape[-1])
                factor = H2.shape[-1]//H1.shape[-1]
                H1new = torch.empty(H1.shape[:-2]+H2.shape[-2:]).to(device)
                h, w = torch.meshgrid(torch.arange(H2.shape[-2]), torch.arange(H2.shape[-1]))
                h, w = h.to(device), w.to(device)
                H1new[..., h, w] = H1[..., h//factor, w//factor]
                H1 = H1new
            if H1.shape[1] != H2.shape[1]:
                H1, H2 = sorted([H1, H2], key = lambda x: x.shape[1])
                padder = torch.zeros((H1.shape[0], H2.shape[1] - H1.shape[1]) + H1.shape[2:]).to(device)
                H1 = torch.cat([H1, padder], dim=1)
            sum1 = torch.linalg.norm(H1, dim=d1, keepdim=True)
            H1 /= (sum1 == 0.0)*1.0 + (sum1 != 0.0)*sum1
            sum2 = torch.linalg.norm(H2, dim=d2, keepdim=True)
            H2 /= (sum2 == 0.0)*1.0 + (sum2 != 0.0)*sum2
            C = torch.tensordot(H1, H2, dims=(d1, d2))
            I, J = spo.linear_sum_assignment(C.cpu().numpy(), maximize=True)
            ans = (H1**2).sum() + (H2**2).sum()  -2*C[I, J].sum()
            ans /= 4*H1.shape[1]
            del H1, H2, C
            return ans.sqrt()
    s = torch.zeros((len(features1), len(features2)))
    for i, H1 in enumerate(features1):
        for j, H2 in enumerate(features2):
            if diagonly and i!=j:
                continue
            s[i, j] = pairwise_procrustes(H1, H2)
    return 1-s

def wreath_cka(features1: List[torch.Tensor],
    features2: List[torch.Tensor], centered=True, 
    estimator='max',
    dims: Tuple[int] = (1, 1), device=torch.device('cpu'),
    progress=True, batched=False):
    print("calculating wreath CKA")
    def kernel(H: torch.Tensor, centered=centered):
        with torch.no_grad():
            H= H.to(device)
            H = H.transpose(1, dims[0])
            d =  [i for i in range(len(H.shape)) if i != 1]
            if centered:
                H -= H.mean(dim=d,keepdim=True)
            nrms = torch.linalg.norm(H, dim=d, keepdim=True)
            H /= (nrms == 0.0)*1.0 + (nrms != 0.0)*nrms

            K = torch.zeros((H.shape[0], H.shape[0])).to(device)
            Kloop = range(1, K.shape[1])
            if progress:
                Kloop = tqdm(Kloop, desc='K')
            for j in Kloop:
                if estimator == 'max':
                    K[:j, j, ...], _ = (H[:j]*H[j]).sum(dim=(-2,-1)).max(dim=1)
                elif estimator == 'median':
                    K[:j, j, ...], _ = (H[:j]*H[j]).sum(dim=(-2,-1)).median(dim=1)
                if progress:
                    Kloop.set_description(f'K nrm: {K.norm()}')
            K += K.clone().transpose(0,1)
        return K.cpu()
    
    def hsic1(K: torch.Tensor, L:torch.Tensor):
        n = K.shape[0]
        o = torch.ones((n,))
        ans = (K*L).sum() +(K.sum()*L.sum())/((n-1)*(n-2)) -  \
            (2/(n-2))*((K.sum(dim=1, keepdim=True))*(L.sum(dim=1,keepdim=True))).sum()
        return ans/(n*(n-3))

    with torch.no_grad():
        Ks, Ls = [[kernel(H) for H in f] for f in [features1, features2]]
        s = torch.empty((len(Ks), len(Ls)))
        for i, K in enumerate(Ks):
            for j, L in enumerate(Ls):
                K, L = K.to(device), L.to(device)
                s[i, j] = hsic1(K, L)/(hsic1(K, K).sqrt()*hsic1(L,L).sqrt())
    return s

def ortho_procrustes(features1: List[torch.Tensor],
    features2: List[torch.Tensor], diagonly=True,
    dims: Tuple[int] = (1, 1), device=torch.device('cpu'), 
    progress=True,batched=False):
    print("calculating ortho procrustes")
    def pairwise_ortho_procrustes(H1: torch.Tensor, H2: torch.Tensor):
        with torch.no_grad():
            H1, H2 = deepcopy(H1), deepcopy(H2)
            H1, H2 = H1.to(device), H2.to(device)
            H1 = H1.transpose(1, dims[0])
            H2 = H2.transpose(1, dims[1])
            d1, d2 = [[i for i in range(len(h.shape)) if i != 1]
                for  h in (H1, H2)]
            if H1.shape[-1] != H2.shape[-1]:
                H1, H2 = sorted([H1, H2], key = lambda x: x.shape[-1])
                factor = H2.shape[-1]//H1.shape[-1]
                H1new = torch.empty(H1.shape[:-2]+H2.shape[-2:]).to(device)
                h, w = torch.meshgrid(torch.arange(H2.shape[-2]), torch.arange(H2.shape[-1]))
                h, w = h.to(device), w.to(device)
                H1new[..., h, w] = H1[..., h//factor, w//factor]
                H1 = H1new
            if H1.shape[1] != H2.shape[1]:
                H1, H2 = sorted([H1, H2], key = lambda x: x.shape[1])
                padder = torch.zeros((H1.shape[0], H2.shape[1] - H1.shape[1]) + H1.shape[2:]).to(device)
                H1 = torch.cat([H1, padder], dim=1)
            sum1 = torch.linalg.norm(H1, keepdim=True)
            H1 /= (sum1 == 0.0)*1.0 + (sum1 != 0.0)*sum1
            sum2 = torch.linalg.norm(H2, keepdim=True)
            H2 /= (sum2 == 0.0)*1.0 + (sum2 != 0.0)*sum2
            C = torch.tensordot(H1, H2, dims=(d1, d2))
            ans = (H1**2).sum() + (H2**2).sum()  -2*torch.linalg.norm(C, ord='nuc')
            del H1, H2, C
            return ans.sqrt()/2
    s = torch.zeros((len(features1), len(features2)))
    for i, H1 in enumerate(features1):
        for j, H2 in enumerate(features2):
            if diagonly and i != j:
                continue
            s[i, j] = pairwise_ortho_procrustes(H1, H2)
    return 1-s

def ortho_cka(features1: List[torch.Tensor],
    features2: List[torch.Tensor], centered=True, 
    dims: Tuple[int] = (1, 1), device=torch.device('cpu'),
    progress=True, batched=False):
    print("calculating ortho CKA")
    def kernel(H: torch.Tensor, centered=centered):
        with torch.no_grad():
            H= H.to(device)
            H = H.transpose(1, dims[0])
            d =  [i for i in range(len(H.shape)) if i != 1]
            if centered:
                H -= H.mean(dim=d,keepdim=True)
            nrms = torch.linalg.norm(H, dim=d, keepdim=True)
            H /= (nrms == 0.0)*1.0 + (nrms != 0.0)*nrms
            K = torch.zeros((H.shape[0], H.shape[0])).to(device)
            Kloop = range(1, K.shape[1])
            if progress:
                Kloop = tqdm(Kloop, desc='K')
            for j in Kloop:
                K[:j, j, ...] = (H[:j]*H[j]).sum(dim=(-3,-2,-1))
                if progress:
                    Kloop.set_description(f'K nrm: {K.norm()}')
            K += K.clone().transpose(0,1)
        return K.cpu()

    def hsic1(K: torch.Tensor, L:torch.Tensor):
        n = K.shape[0]
        o = torch.ones((n,))
        ans = (K*L).sum() +(K.sum()*L.sum())/((n-1)*(n-2)) -  \
            (2/(n-2))*((K.sum(dim=1, keepdim=True))*(L.sum(dim=1,keepdim=True))).sum()
        return ans/(n*(n-3))

    with torch.no_grad():
        Ks, Ls = [[kernel(H) for H in f] for f in [features1, features2]]
        s = torch.empty((len(Ks), len(Ls)))
        for i, K in enumerate(Ks):
            for j, L in enumerate(Ls):
                K, L = K.to(device), L.to(device)
                s[i, j] = hsic1(K, L)/(hsic1(K, K).sqrt()*hsic1(L,L).sqrt())
    return s