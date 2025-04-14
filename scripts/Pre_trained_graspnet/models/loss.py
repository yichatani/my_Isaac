import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

# load model config
with open('./models/model_config.yaml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

Loss_Weights = {
    'objectness': 1,
    'graspness': 10,
    'sealness': 1,
    'wrenchness': 10,
    'flatness': 10,
    'view': 100,
    'score': 15,
    'width': 10,
}

def get_loss(end_points, data_type):
    # common objectness loss
    objectness_loss, end_points = compute_objectness_loss(end_points)
    # branches
    if model_config['Global']['training_branch'] == 'grasp':
        if data_type == 'graspnet':
            graspness_loss, end_points = compute_graspness_loss(end_points)
            view_loss, end_points = compute_view_graspness_loss(end_points)
            score_loss, end_points = compute_score_loss(end_points)
            width_loss, end_points = compute_width_loss(end_points)
            loss = objectness_loss + graspness_loss + view_loss + score_loss + width_loss
        elif data_type == 'meta':
            score_loss, end_points = compute_score_loss(end_points)
            width_loss, end_points = compute_width_loss(end_points)
            loss = objectness_loss  + score_loss + width_loss
    elif model_config['Global']['training_branch'] == 'suction':
        suctioness_loss, end_points = compute_suctioness_loss(end_points)
        loss = objectness_loss + suctioness_loss
    elif model_config['Global']['training_branch'] == 'both':
        graspness_loss, end_points = compute_graspness_loss(end_points)
        # suctioness_loss, end_points = compute_suctioness_loss(end_points)
        view_loss, end_points = compute_view_graspness_loss(end_points)
        score_loss, end_points = compute_score_loss(end_points)
        width_loss, end_points = compute_width_loss(end_points)
        loss = objectness_loss + graspness_loss + view_loss + score_loss + width_loss
        # loss = objectness_loss  + view_loss + score_loss + width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points



def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    loss = criterion(objectness_score, objectness_label)
    loss *= Loss_Weights['objectness']
    end_points['loss/stage1_objectness_loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()
    return loss, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1) # (B, N)
    graspness_label = end_points['graspness_label'].squeeze(-1) # (B, N)

    loss_mask = end_points['objectness_label'].bool()
    graspness_loss = criterion(graspness_score, graspness_label)
    graspness_loss = graspness_loss[loss_mask].mean() * Loss_Weights['graspness']
    
    graspness_score_c = graspness_score.detach().clone()[loss_mask]
    graspness_label_c = graspness_label.detach().clone()[loss_mask]
    graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
    graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    end_points['stage1_graspness_acc_rank_error'] = rank_error

    end_points['loss/stage1_graspness_loss'] = graspness_loss
    return graspness_loss, end_points


def compute_suctioness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    sealness_score = end_points['sealness_score'].squeeze(1) # (B, N)
    sealness_label = end_points['sealness_label'].squeeze(-1) # (B, N)
    wrenchness_score = end_points['wrenchness_score'].squeeze(1) # (B, N)
    wrenchness_label = end_points['wrenchness_label'].squeeze(-1) # (B, N)
    # flatness_score = end_points['flatness_score'].squeeze(1) # (B, N)
    # flatness_label = end_points['flatness_label'].squeeze(-1) # (B, N)

    loss_mask = end_points['objectness_label'].bool()
    sealness_loss = criterion(sealness_score, sealness_label)
    sealness_loss = sealness_loss[loss_mask].mean() * Loss_Weights['sealness']
    wrenchness_loss = criterion(wrenchness_score, wrenchness_label)
    wrenchness_loss = wrenchness_loss[loss_mask].mean() * Loss_Weights['wrenchness']
    # flatness_loss = criterion(flatness_score, flatness_label)
    # flatness_loss = flatness_loss[loss_mask].mean() * Loss_Weights['flatness']
    # suctioness_loss = sealness_loss + wrenchness_loss + flatness_loss
    suctioness_loss = sealness_loss + wrenchness_loss

    end_points['loss/stage1_sealness_loss'] = sealness_loss
    end_points['loss/stage1_wrenchness_loss'] = wrenchness_loss
    # end_points['loss/stage1_flatness_loss'] = flatness_loss
    return suctioness_loss, end_points


def compute_view_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)
    loss *= Loss_Weights['view']
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points


def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)
    loss *= Loss_Weights['score']
    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    
    loss_mask = grasp_score_label > 0
    # print('loss_mask', np.where(loss_mask.detach().cpu().numpy() != False))
    loss = loss[loss_mask].mean()
    loss *= Loss_Weights['width']
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points
