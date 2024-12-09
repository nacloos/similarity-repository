import torch
import torch.nn.functional as F
from metrics.LayerWiseMetrics import patience_loss

def estimate_metrics(teacher_featuremap, student_featuremap, args):

    pkd_loss= 0.0
    length = len(teacher_featuremap)
    for i in range(length-1):
        pkd_loss += patience_loss(torch.flatten(teacher_featuremap[i], start_dim=-2, end_dim=-1),
                                  torch.flatten(student_featuremap[i], start_dim=-2, end_dim=-1 ))

    inter_dist_loss = inter_dist(teacher_featuremap[-1], student_featuremap[-1], args)
    inter_cos_loss = inter_cos(teacher_featuremap[-1], student_featuremap[-1], args)
    intra_dist_loss = intra_dist(teacher_featuremap[-1], student_featuremap[-1])
    intra_cos_loss = intra_cos(teacher_featuremap[-1], student_featuremap[-1])

    return pkd_loss, inter_dist_loss, inter_cos_loss, intra_dist_loss, intra_cos_loss

def inter_dist(teacher, student, args):
    L1 = torch.nn.L1Loss(reduction='mean')

    new_teacher = teacher.view(args.per_gpu_train_batch_size, -1)
    new_student = student.view(args.per_gpu_train_batch_size, -1)

    teacher_dist = torch.cdist(new_teacher, new_teacher)
    student_dist = torch.cdist(new_student, new_student)
    exter_dist_loss = L1(student_dist, teacher_dist)

    teacher_mean = torch.mean(teacher_dist)
    student_mean = torch.mean(student_dist)
    teacher_std = torch.std(teacher_dist)
    student_std = torch.std(student_dist)

    return [exter_dist_loss, teacher_mean, teacher_std, student_mean, student_std]

def inter_cos(teacher, student, args):
    L1 = torch.nn.L1Loss()

    new_teacher = F.normalize(teacher.view(args.per_gpu_train_batch_size, -1), dim=-1)
    new_student = F.normalize(student.view(args.per_gpu_train_batch_size, -1), dim=-1)
    teacher_cos = torch.mm(new_teacher, new_teacher.transpose(-1,-2))
    student_cos = torch.mm(new_student, new_student.transpose(-1, -2))
    exter_cos_loss = L1(student_cos, teacher_cos)

    teacher_abs_cos = torch.abs(teacher_cos)
    student_abs_cos = torch.abs(student_cos)

    teacher_mean = torch.mean(teacher_abs_cos)
    student_mean = torch.mean(student_abs_cos)
    teacher_std = torch.std(teacher_abs_cos)
    student_std = torch.std(student_abs_cos)

    return [exter_cos_loss, teacher_mean, teacher_std, student_mean, student_std]

def intra_dist(teacher, student):
    L1 = torch.nn.L1Loss()

    teacher_dist = torch.cdist(teacher, teacher)
    student_dist = torch.cdist(student, student)
    inter_dist_loss = L1(student_dist, teacher_dist)

    teacher_mean = torch.mean(teacher_dist)
    student_mean = torch.mean(student_dist)
    teacher_std = torch.std(teacher_dist)
    student_std = torch.std(student_dist)

    return [inter_dist_loss, teacher_mean, teacher_std, student_mean, student_std]

def intra_cos(teacher, student):
    L1 = torch.nn.L1Loss()

    new_teacher = F.normalize(teacher, dim=-1)
    new_student = F.normalize(student, dim=-1)

    teacher_cos = torch.bmm(new_teacher, new_teacher.transpose(-1, -2))
    student_cos = torch.bmm(new_student, new_student.transpose(-1 ,-2))
    inter_cos_loss = L1(student_cos, teacher_cos)

    teacher_abs_cos = torch.abs(teacher_cos)
    student_abs_cos = torch.abs(student_cos)

    teacher_mean = torch.mean(teacher_abs_cos)
    student_mean = torch.mean(student_abs_cos)
    teacher_std = torch.std(teacher_abs_cos)
    student_std = torch.std(student_abs_cos)

    return [inter_cos_loss, teacher_mean, teacher_std, student_mean, student_std]