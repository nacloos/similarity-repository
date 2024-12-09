import torch
import os
import tqdm
import numpy as np
import torch.nn as nn
import networkx as nx
import heapq
# from scipy.stats import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import data
from train import get_trainer, get_dataset, get_model, set_gpu
from metric.motifs import ESU_BFS

cfgs = {
'cresnet18': [2, 2, 2, 2],
'cresnet34': [3, 4, 6, 3],
'cresnet50': [3, 4, 6, 3],
'cresnet101': [3, 4, 23, 3],
'cresnet152': [3, 8, 36, 3],
'cvgg11_bn': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22',
              'features.25'],
'cvgg13_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21',
              'features.24', 'features.28', 'features.31'],
'cvgg16_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
              'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
'cvgg19_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
              'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40',
              'features.43',
              'features.46', 'features.49'],
}


def get_inner_feature(model, hook, arch='cres18'):
    cfg = cfgs[arch]
    print('cfg:', cfg)
    if 'res' in arch:
        handle = model.conv1.register_forward_hook(hook)
        # handle.remove()  # free memory
        for i in range(cfg[0]):
            handle = model.layer1[i].register_forward_hook(hook)

        for i in range(cfg[1]):
            handle = model.layer2[i].register_forward_hook(hook)

        for i in range(cfg[2]):
            handle = model.layer3[i].register_forward_hook(hook)

        for i in range(cfg[3]):
            handle = model.layer4[i].register_forward_hook(hook)
    elif 'vgg' in arch:
        count = 0
        for idx, m in enumerate(model.named_modules()):
            name, module = m[0], m[1]
            if count < len(cfg):
                if name == cfg[count]:
                    print(module)
                    handle = module.register_forward_hook(hook)
                    count += 1
            else:
                break


def Cosine_Similarity(x, y):
    r'''
    Args:
        x: feature
        y: feature
    Returns: the similarity between x and y
    '''
    return torch.cosine_similarity(x, y)


def calculate_cosine_similarity_matrix(h_emb, topk, eps=1e-8):
    r'''
        h_emb: (N, M) hidden representations
    '''
    # normalize
    edges_list = []

    h_emb = torch.tensor(h_emb)
    batch_size = h_emb.size(0)
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    sim_matrix = sim_matrix.cpu().numpy()
    row, col = np.diag_indices_from(sim_matrix)  # Do not consider self-similarity
    sim_matrix[row, col] = 0
    n_samples = h_emb.shape[0]

    for i in range(n_samples):  # note that: nodes start from 1
        k_indice = heapq.nlargest(topk, range(len(sim_matrix[i])), sim_matrix[i].take)
        for j in range(len(k_indice)):
            b = int(k_indice[j]+1)
            a = (int(i+1), b, float(sim_matrix[i][k_indice[j]]))
            edges_list.append(a)
    sim_matrix = nx.from_numpy_matrix(sim_matrix)
    return sim_matrix, edges_list


def LSim(G1, G2):
    r'''

    Args:
        G1: undirected graph
        G2: undirected graph

    Returns: layer similarity, the similarity of nodes between layer a and b
    ref: 'Measuring similarity for clarifying layer difference in multiplex ad hoc duplex information networks' Page 3
    '''
    adj1 = nx.adjacency_matrix(G1).todense()
    adj2 = nx.adjacency_matrix(G2).todense()
    lsim = 0
    NSim = []
    nodes = nx.number_of_nodes(G1)
    for i in range(nodes):
        k_i_1 = adj1[i]
        k_i_2 = adj2[i]
        NSim_i = cosine_similarity(k_i_1, k_i_2)
        NSim.append(NSim_i)
        lsim += NSim_i / nodes

    return lsim, NSim


def get_graphs(args, adj=False):
    model_graphs = []
    if 'vgg' in args.arch.lower():
        my_seeds = [23, 24, 26, 257, 277, 287, 298, 300, 31, 32]
        model_size = 13
    elif 'cnn' in args.arch.lower():
        my_seeds = range(23, 33)
        model_size = 8
    else:
        my_seeds = range(23, 33)
        model_size = 9  # res18

    for s in my_seeds:
        undirected_G_list = []
        for layer in range(model_size):
            file_check = os.path.join(args.feature_save, "{arch}-{layer}-sd{seed}-N{de}_{metric}.npy".format(
                arch=args.arch, layer=layer, seed=s, de=args.batch_size, metric=args.metric))

            feature = np.load(file_check)
            weighted_network, adj_mat = calculate_cosine_similarity_matrix(feature, args.topk)
            if not adj:
                undirected_G_list.append(weighted_network)
            else:
                undirected_G_list.append(nx.from_numpy_matrix(adj_mat))
        model_graphs.append(undirected_G_list)
    return model_graphs


def sanity_check(graph_save, args):
    model_graphs = get_graphs(graph_save, args)
    model_num = len(model_graphs)
    total_acc = []
    for i in range(model_num):
        for j in range(i+1, model_num):  # use i_th model to find the corresponding layer of the rest j models
            # above is the iteration of models, below is the iteration of layers
            current_layer = 0
            correct_cnt = 0
            for layer_i in model_graphs[i]:
                sim_of_layers = []
                for layer_j in model_graphs[j]:
                    sim_of_layers.append(LSim(layer_i, layer_j))
                found_layer = sim_of_layers.index(max(sim_of_layers))
                if found_layer == current_layer:
                    correct_cnt +=1
                current_layer += 1
            acc = correct_cnt/current_layer
            total_acc.append(acc)
            print("model {i} and model {j}, acc is {acc}".format(i=i, j=j, acc=acc))
    print("total acc is {ta}".format(ta=np.mean(total_acc)))
    return np.mean(total_acc)


def func_sim(args):
    train, validate, modifier = get_trainer(args)
    data = get_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda()
    info = {'set': 'cifar10', 'bs': 200, 'topk': 5, 'acc1': 0}

    model = get_model(args)
    model = set_gpu(args, model)
    acc1, acc5 = validate(
        data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
    )
    info['acc1'] = acc1
    save_dir = './graph_save/function'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    device = torch.device("cuda:{}".format(args.gpu))
    get_edge_list(model, data.val_loader, save_dir, device, arch=args.arch, topk=5, degree=200)
    get_motif(save_dir, arch=args.arch, log_dir='./', log_info=info)


def get_motif(graph_save, arch, log_dir, log_info):
    '''

    :param graph_save: where saves graphs file
    :param arch: model structure
    :param log_dir: where to save log file
    :param log_info: dict e.g., {'set': 'cifar10', 'bs': 128, 'topk': 5, 'acc1': 00.0}
    :return:
    '''
    if 'res' in arch:
        layer_num = sum(cfgs[arch])
    elif 'vgg' in arch:
        layer_num = len(cfgs[arch])
    file_path = os.path.join(log_dir, 'motif_log.csv')
    logfile = open(file_path, 'a')  # vgg
    logfile.flush()
    if os.path.getsize(file_path) == 0:
        logfile.write('arch, set, batch_size, topk, layer, acc1, type1, type2, type3\n')
        logfile.flush()
    label_check = os.path.join(graph_save, "{arch}-N{de}_label.npy".format(arch=arch, de=log_info['bs']))
    label = np.load(label_check)
    for m in range(layer_num):
        file_check = os.path.join(graph_save,
                                  "{arch}-{layer}-top{k}-N{de}_edge.npy".format(arch=arch, layer=m,
                                                                                k=log_info['topk'], de=log_info['bs']))
        edge = np.load(file_check)
        graph, adj = graph_from_edges(edge, label, log_info['bs'])
        motif = ESU_BFS(adj, label)
        logfile.write('{arch}, {set}, {batch_size}, {topk}, {layer}, {acc1}, {type1}, {type2}, {type3}\n'.format(
            arch=arch, set=log_info['set'], batch_size=log_info['bs'], topk=log_info['topk'], layer=m, acc1=log_info['acc1'],
            type1=motif.type1, type2=motif.type2, type3=motif.type3
        ))
        logfile.flush()


def graph_from_edges(edges_list, labels, batch_size):  # checked
    r'''

    Args:
        edges_list: direcetd weighted networks' edges list, a tuple like: (source node, target node, weight)
        whole_label: a list of all samples' ground true & predicted top5 labels
        label: ground true label
    Returns: undirected weighted network, the adj of undirected weighted network

    '''
    adj = np.zeros((batch_size, batch_size))

    # directed adj
    for ii in range(len(edges_list)):
        a, b = int(edges_list[ii][0] - 1), int(edges_list[ii][1] - 1)
        adj[a, b] = edges_list[ii][2]

    adj = (adj + adj.T) / 2

    undirected_weighted_network = nx.from_numpy_matrix(adj)
    for i in range(batch_size):
        undirected_weighted_network.nodes[i]['label'] = labels[i]

    return undirected_weighted_network, adj


def get_edge_list(model, val_loader, graph_save, device, arch, topk=5, degree=200):
    if not os.path.exists(graph_save):
        os.makedirs(graph_save)

    for i, (images, target) in tqdm.tqdm(enumerate(val_loader), ascii=True, total=len(val_loader)):
        inter_feature = []

        def hook(module, input, output):
            inter_feature.append(output.clone().detach())

        model.eval()
        images = images.to(device)
        target = target.to(device)
        with torch.no_grad():
            get_inner_feature(model, hook, arch=arch.lower())
            output = model(images)
            label_check = os.path.join(graph_save, "{arch}-N{de}_label.npy".format(arch=arch, de=degree))
            np.save(label_check, target.cpu().detach().numpy())
            for m in range(len(inter_feature)):
                if len(inter_feature[m].shape) != 2:
                    inter_feature[m] = inter_feature[m].reshape(degree, -1)
                file_check = os.path.join(graph_save,
                        "{arch}-{layer}-top{k}-N{de}_edge.npy".format(arch=arch, layer=m, k=topk, de=degree))

                if os.path.exists(file_check):
                    print(file_check + " ****exist skip**** !")
                    continue
                else:
                    similarity_matrix, edges_list = calculate_cosine_similarity_matrix(inter_feature[m], topk)
                    np.save(file_check, edges_list)
                    print(file_check + " saved !")
        break  # one batch is enough