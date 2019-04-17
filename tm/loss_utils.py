import torch.nn as nn
import torch
import sys
def compute_losses(targets, all_confs, all_f_cls):
    # each loss can be computed independently, and only when you backprop it should you make sure you compute the gradient
    # only on the corresponding weights
    f_cls_losses = compute_f_cls_losses_for_f_cls(targets, all_f_cls)
    f_cls_losses_for_conf_eval = compute_f_cls_loss_for_conf_eval(targets, all_f_cls)
    conf_eval_losses = compute_conf_eval_losses(all_confs, f_cls_losses_for_conf_eval)
    q_m_losses = compute_q_m_losses(all_confs)
    a_m_losses = compute_a_m_losses(all_confs)
    return conf_eval_losses, \
           f_cls_losses, \
           q_m_losses, \
           a_m_losses


def compute_f_cls_losses(targets, all_f_cls, criterion, repeat_label = False):
    # todo Each loss should be between 0 and 1, in order to easily train the conf eval. Find the right criterion

    losses = []

    for label, predicted_labels in zip(list(targets), all_f_cls):
        if repeat_label:            
            label = label.repeat(len(predicted_labels))

        losses.append(criterion(predicted_labels, label))

    return losses

def compute_f_cls_losses_for_f_cls(targets, all_f_cls):
    criterion = nn.CrossEntropyLoss(reduction='none')
    return compute_f_cls_losses(targets, all_f_cls, criterion, repeat_label=True)

def compute_f_cls_loss_for_conf_eval(targets, all_f_cls):
    conf_eval_losses = []
    for label, predicted_labels in zip(list(targets), all_f_cls):
        values_of_correct_label = predicted_labels[:, label]
        conf_eval_losses.append(values_of_correct_label)
    return conf_eval_losses

def compute_conf_increase_loss(all_conf):
    # the target is to increase the confidence at each iteration
    losses = []

    for confs_of_one_sample in all_conf:
        current_sample_losses = []

        prev_conf = 1.0
        for conf in confs_of_one_sample:
            current_sample_losses.append((prev_conf - conf).unsqueeze(0))
            prev_conf = conf

        losses.append(torch.cat(current_sample_losses))

    return losses


def compute_q_m_losses(all_conf):
    return compute_conf_increase_loss(all_conf)


def compute_a_m_losses(all_conf):
    return compute_conf_increase_loss(all_conf)


def compute_conf_eval_losses(all_conf, all_f_cls_outputs):
    # the target of the confidence evaluator is to predict the error of the final classifier.
    # Because the name is 'confidence' , it should predict the inverse of this
    # because a high value would mean a high error, which means a very low confidence
    # todo either change from confidence evaluator to lack-of-confidence evaluator, or find the right way to formulate the loss
    losses = []
    criterion = nn.L1Loss(reduction='none')
    for current_sample_confs, f_cls_output in zip(all_conf, all_f_cls_outputs):
        losses.append(criterion(current_sample_confs, f_cls_output))
    return losses






