import torch
import argparse


class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)


def main():
    ### SET UP ARG PARSER ###

    parser = argparse.ArgumentParser(description='KMeans')
    parser.add_argument('--backbone-a', default='', type=str, help='backbone path')
    parser.add_argument('--backbone-b', default='', type=str, help='backbone path')
    parser.add_argument('--dataset', default='imagenet', type=str, help='datset name: cub or imagenet')
    args = parser.parse_args()

    ##########################

    features_a = torch.load('/vulcanscratch/mgwillia/vissl/cka_features/' + '_'.join([args.backbone_a, args.dataset, 'features']) + '.pth.tar')
    features_b = torch.load('/vulcanscratch/mgwillia/vissl/cka_features/' + '_'.join([args.backbone_b, args.dataset, 'features']) + '.pth.tar')
    
    device = torch.device('cuda')
    cuda_cka = CudaCKA(device)

    for layer_name, representation_a in features_a.items():
        representation_b = features_b[layer_name]
        if args.dataset == 'imagenet':
            cur_CKA = cuda_cka.linear_CKA(representation_a[::5].cuda(), representation_b[::5].cuda())
        else:
            cur_CKA = cuda_cka.linear_CKA(representation_a.cuda(), representation_b.cuda())
        print(f'Dataset: {args.dataset}, Backbone_A: {args.backbone_a}, Backbone_B: {args.backbone_b}, Layer: {layer_name}, Linear_CKA: {cur_CKA}', flush=True)


if __name__ == '__main__':
    main()
