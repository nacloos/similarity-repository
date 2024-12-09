import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb
from scipy import stats


# reference: github.com/yuanli2333/CKA-Centered-Kernel-Aligment/blob/master/cca_core.py
# omit the code
def centering(K, args):
    n = K.size()[-1]

    unit = torch.ones(size=(n, n)).to(args.device)
    I = torch.eye(n).to(args.device)
    H = I - unit / n

    return torch.matmul(torch.matmul(H, K), H)


def linear_HSIC(X, Y, args):
    if X.dim() >= 3 and Y.dim() >= 3:
        L_X = torch.matmul(X, X.transpose(-2, -1))
        L_Y = torch.matmul(Y, Y.transpose(-2, -1))

        return torch.sum(torch.sum(torch.mul(centering(L_X, args), centering(L_Y, args)), dim=-1), dim=-1)
    else:
        L_X = torch.matmul(X, X.transpose(-2, -1))
        L_Y = torch.matmul(Y, Y.transpose(-2, -1))
        square_x = centering(L_X, args)
        square_y = centering(L_Y, args)
        output = torch.trace(torch.mm(square_x, square_y)) / (X.size(-2) ** 2 - 1)
        return output
        # return torch.sum(torch.mul(centering(L_X, args), centering(L_Y, args)))


def linear_CKA_loss(X, Y, args):
    hsic = linear_HSIC(X, Y, args)
    var1 = torch.sqrt(linear_HSIC(X, X, args))
    var2 = torch.sqrt(linear_HSIC(Y, Y, args))
    # one = torch.tensor([1]).to(args.device)
    return -torch.log(torch.mean(torch.abs(torch.div(hsic, (var1 * var2)))) + 1e-8)


def cdist2(x, y, eps=1e-8):
    if y.ndim == 1:
        y = y.view(1, -1)
    x_sq_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_sq_norm = y.pow(2).sum(dim=-1)
    if y_sq_norm.ndim == 1:
        y_sq_norm = y_sq_norm.unsqueeze(0)
    x_dot_y = x @ y.transpose(-1, -2) if len(y) > 1 else x @ y.transpose(0, 1)
    sq_dist = x_sq_norm + y_sq_norm.unsqueeze(dim=-2) - 2 * x_dot_y
    sq_dist.clamp_(min=0.0)
    return torch.sqrt(sq_dist + eps)


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def patience_loss(teacher_patience, student_patience, normalized_patience=True):
    if normalized_patience:
        teacher_patience = F.normalize(teacher_patience, p=2, dim=-1)
        student_patience = F.normalize(student_patience, p=2, dim=-1)
    return F.mse_loss(student_patience.float(), teacher_patience.float()).half()


class CKA_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.internal = args.internal

    def forward(self, teacher, student, args, length=5):
        cka_loss = 0.0
        if self.internal:
            for i in range(length):
                cka_loss += linear_CKA_loss(teacher[i], student[i], args)
        else:
            for i in range(length):
                teacher_featuremaps = teacher[i].view(args.per_gpu_train_batch_size, -1)
                student_featuremaps = student[i].view(args.per_gpu_train_batch_size, -1)
                cka_loss += linear_CKA_loss(teacher_featuremaps, student_featuremaps, args)
        cka_loss /= length

        return cka_loss


# RKD reference
def RKDAngle(teacher, student, args):
    # N x C
    # N x N x C
    teacher, student = teacher[0].view(args.per_gpu_train_batch_size, -1), student[0].view(
        args.per_gpu_train_batch_size, -1)
    # with torch.no_grad():
    td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
    norm_td = F.normalize(td, p=2, dim=2)
    t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (student.unsqueeze(0) - student.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss = F.smooth_l1_loss(s_angle, t_angle)
    return loss


def RkDDistance(teacher, student, args):
    teacher, student = teacher[0].view(args.per_gpu_train_batch_size, -1), student[0].view(
        args.per_gpu_train_batch_size, -1)
    # with torch.no_grad():
    t_d = pdist(teacher, squared=False)
    mean_td = t_d[t_d > 0].mean()
    t_d = t_d / mean_td

    d = pdist(student, squared=False)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss = F.smooth_l1_loss(d, t_d)
    return loss


class CKA(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, teacher, student, args):
        cka_loss = linear_CKA_loss(teacher, student, args)
        return cka_loss


class CKA_mixed_loss(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, teacher, student, args, length=5):
        cka_inter_loss, cka_exter_loss = 0.0, 0.0

        for i in range(length):
            cka_inter_loss += linear_CKA_loss(teacher[i], student[i], args)
            cka_exter_loss += linear_CKA_loss(teacher[i].view(args.per_gpu_train_batch_size, -1),
                                              student[i].view(args.per_gpu_train_batch_size, -1),
                                              args)
        cka_inter_loss /= length
        cka_exter_loss /= length

        return cka_inter_loss, cka_exter_loss


class PKD_loss(nn.Module):
    def forward(self, teacher, student, args, length=5, normalized_patience=True):
        pkd_loss = 0.0
        for i in range(length):
            pkd_loss += patience_loss(torch.flatten(teacher[i], start_dim=-2, end_dim=-1),
                                      torch.flatten(student[i], start_dim=-2, end_dim=-1), normalized_patience)

        return pkd_loss


class Internal_loss(nn.Module):
    def forward(self, teacher, student, length):
        internal_distance_loss = 0.0
        internal_cosine_loss = 0.0

        mse_loss = nn.MSELoss()
        cos_loss = nn.CosineSimilarity(dim=-1)
        for i in range(length):
            teacher_featuremap = teacher[3][2 * (i + 1)]
            student_featuremap = student[3][i + 1]
            # euclidian distance
            teacher_euclidean = cdist2(teacher_featuremap, teacher_featuremap)
            student_euclidean = cdist2(student_featuremap, student_featuremap)

            # normalize T & S for cosine similarity
            norm_teacher_featuremap = F.normalize(teacher_featuremap, dim=-1)
            norm_student_featuremap = F.normalize(student_featuremap, dim=-1)

            # cosine similarity
            teacher_uncen_cosine = torch.bmm(norm_teacher_featuremap,
                                             norm_teacher_featuremap.transpose(-1, -2))
            student_uncen_cosine = torch.bmm(norm_student_featuremap,
                                             norm_student_featuremap.transpose(-1, -2))

            internal_distance_loss += mse_loss(teacher_euclidean, student_euclidean)
            internal_cosine_loss += mse_loss(teacher_uncen_cosine, student_uncen_cosine)

        internal_distance_loss /= length
        internal_cosine_loss /= length

        return internal_distance_loss, internal_cosine_loss


class External_1D_loss(nn.Module):
    # matching centroid of features
    # get euclidean distance and consine similarity
    def forward(self, teacher, student, length):
        external_centroid_distance_loss = 0.0
        external_centroid_cosine_loss = 0.0

        mse_loss = nn.MSELoss()
        cos_loss = nn.CosineSimilarity(dim=-1)

        for i in range(length):
            teacher_featuremap = teacher[3][2 * (i + 1)]
            student_featuremap = student[3][i + 1]

            external_centroid_distance_loss += mse_loss(teacher_featuremap, student_featuremap)

            teacher_featuremap = torch.flatten(teacher_featuremap)
            student_featuremap = torch.flatten(student_featuremap)

            external_centroid_cosine_loss -= torch.log(torch.abs(cos_loss(teacher_featuremap, student_featuremap)))

        external_centroid_cosine_loss /= length
        external_centroid_distance_loss /= length

        return external_centroid_distance_loss, external_centroid_cosine_loss


class External_structure(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.teacher_initialize = torch.zeros(size=(config.num_hidden_layers - 1, args.num_batch_group,
                                                    config.hidden_size * args.max_seq_length)).to(args.device)
        self.student_initialize = torch.zeros(size=(config.num_hidden_layers - 1, args.num_batch_group,
                                                    config.hidden_size * args.max_seq_length)).to(args.device)

    def forward(self, teacher, student, length):
        teacher_centroid, student_centroid = [], []

        mse_loss = nn.MSELoss()
        # extract featuremap of each hidden layers
        for i in range(length):
            teacher_centroid.append(torch.mean(teacher[3][i], dim=0).view(1, -1))
            student_centroid.append(torch.mean(student[3][i], dim=0).view(1, -1))

        teacher_output = torch.stack(teacher_centroid)
        student_output = torch.stack(student_centroid)

        self.teacher_initialize = torch.cat([self.teacher_initialize, teacher_output], dim=1)
        self.student_initialize = torch.cat([self.student_initialize, student_output], dim=1)
        self.teacher_initialize = torch.transpose(self.teacher_initialize, 0, 1)[1:]
        self.student_initialize = torch.transpose(self.student_initialize, 0, 1)[1:]
        # ( # of hidden layers, # of batch group, seq * dim)
        self.teacher_initialize = torch.transpose(self.teacher_initialize, 0, 1)
        self.student_initialize = torch.transpose(self.student_initialize, 0, 1)

        external_teacher_euclidean = cdist2(self.teacher_initialize, self.teacher_initialize)
        external_student_euclidean = cdist2(self.student_initialize, self.student_initialize)

        norm_teahcer = F.normalize(self.teacher_initialize, dim=-1)
        norm_student = F.normalize(self.student_initialize, dim=-1)

        external_teacher_cosine = torch.bmm(norm_teahcer,
                                            norm_teahcer.transpose(-1, -2))
        external_student_cosine = torch.bmm(norm_student,
                                            norm_student.transpose(-1, -2))

        external_structure_distance_loss = mse_loss(external_teacher_euclidean,
                                                    external_student_euclidean)
        external_structure_cosine_loss = mse_loss(external_teacher_cosine,
                                                  external_student_cosine)

        return external_structure_distance_loss, external_structure_cosine_loss


class External_2D_loss(nn.Module):

    def forward(self, teacher, student, length):
        external_distance_loss, external_cosine_loss = 0.0, 0.0

        mse_loss = nn.MSELoss()

        for i in range(length):
            teacher_featuremap = torch.flatten(teacher[3][2 * (i + 1)], start_dim=-2, end_dim=-1)
            student_featuremap = torch.flatten(student[3][i + 1], start_dim=-2, end_dim=-1)

            # euclidian distance
            teacher_euclidean = cdist2(teacher_featuremap, teacher_featuremap)
            student_euclidean = cdist2(student_featuremap, student_featuremap)

            # normalize T & S for cosine similarity
            norm_teacher_featuremap = F.normalize(teacher_featuremap, dim=-1)
            norm_student_featuremap = F.normalize(student_featuremap, dim=-1)

            # cosine similarity
            teacher_cosine = torch.mm(norm_teacher_featuremap,
                                      norm_teacher_featuremap.transpose(-1, -2))
            student_cosine = torch.mm(norm_student_featuremap,
                                      norm_student_featuremap.transpose(-1, -2))

            external_distance_loss += mse_loss(teacher_euclidean, student_euclidean)
            external_cosine_loss += mse_loss(teacher_cosine, student_cosine)

        external_distance_loss /= length
        external_cosine_loss /= length

        return external_distance_loss, external_cosine_loss

class all_structure_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.internal = Internal_loss()
        self.centroid = External_1D_loss()
        self.external = External_2D_loss()

    def forward(self, teacher, student, length):
        internal_distance_loss, internal_cosine_loss = self.internal(teacher, student, length)
        centroid_distance_loss, centroid_cosine_loss = self.centroid(teacher, student, length)
        external_distance_loss, external_cosine_loss = self.external(teacher, student, length)

        loss = (internal_distance_loss, internal_cosine_loss, centroid_distance_loss, centroid_cosine_loss,
                external_distance_loss, external_cosine_loss)
        return loss