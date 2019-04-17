from tm.thinking_machine import TM


def _set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def backward(net: TM, conf_eval_losses, final_classifier_losses, q_m_losses, a_m_losses):
    # Confidence evaluator: TODO
    _set_requires_grad(net, False)
    _set_requires_grad(net.conf_eval, True)

    total_conf_eval_loss = sum(loss.mean() for loss in conf_eval_losses)/len(conf_eval_losses)
    total_conf_eval_loss.backward(retain_graph=True)

    # Final classifier: should only use the loss at the last layer for each sample
    _set_requires_grad(net, False)
    _set_requires_grad(net.f_cls, True)
    _set_requires_grad(net.base_module, True)

    total_final_classifier_loss = sum(fc_loss[-1] for fc_loss in final_classifier_losses)/len(final_classifier_losses)

    total_final_classifier_loss.backward(retain_graph=True)

    # Query machine: TODO explain
    # Answer machine:
    _set_requires_grad(net, True)

    total_q_m_loss = sum(loss.mean() for loss in q_m_losses)/len(q_m_losses)
    total_a_m_loss = sum(loss.mean() for loss in a_m_losses)/len(a_m_losses)

    total_q_m_loss.backward(retain_graph=True)
    total_a_m_loss.backward(retain_graph=True)

    return total_conf_eval_loss.item(), total_final_classifier_loss.item(), total_a_m_loss.item(), total_q_m_loss.item()
