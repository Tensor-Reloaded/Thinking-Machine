import torch.nn as nn


def compute_losses(outputs, all_confs, all_f_cls):
    # each loss can be computed independently, and only when you backprop it should you make sure you compute the gradient
    # only on the corresponding weights

    f_cls_losses = compute_f_cls_losses(outputs, all_f_cls)
    conf_eval_losses = compute_conf_eval_losses(all_confs, f_cls_losses)
    q_m_losses = compute_q_m_losses(all_confs)
    a_m_losses = compute_a_m_losses(all_confs)
    return conf_eval_losses, \
           f_cls_losses, \
           q_m_losses, \
           a_m_losses


def compute_f_cls_losses(outputs, all_f_cls):
    # todo Each loss should be between 0 and 1, in order to easily train the conf eval. Find the right criterion

    losses = []
    # set reduction to none to compute loss between the true label and each other predicted label, not the averege of them
    criterion = nn.MSELoss(reduction='none')

    for label, predicted_labels in zip(list(outputs), all_f_cls):
        losses.append(criterion(label, predicted_labels))

    return losses


def compute_conf_increase_loss(all_conf):
    # the target is to increase the confidence at each iteration
    losses = []

    for confs_of_one_sample in all_conf:
        current_sample_losses = []

        prev_conf = 1.0
        for conf in confs_of_one_sample:
            current_sample_losses.append(prev_conf - conf)
            prev_conf = conf

        losses.append(current_sample_losses)

    return losses


def compute_q_m_losses(all_conf):
    return compute_conf_increase_loss(all_conf)


def compute_a_m_losses(all_conf):
    return compute_conf_increase_loss(all_conf)


def compute_conf_eval_losses(all_conf, f_cls_losses):
    # the target of the confidence evaluator is to predict the error of the final classifier.
    # Because the name is 'confidence' , it should predict the inverse of this
    # because a high value would mean a high error, which means a very low confidence
    # todo either change from confidence evaluator to lack-of-confidence evaluator, or find the right way to formulate the loss
    pass







