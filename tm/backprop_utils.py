from tm.thinking_machine import TM


class BackpropManager:
    def __init__(self, net:TM,
                 cls_patience, cls_eps, cls_max_batch_count,
                 conf_eval_patience, conf_eval_eps, conf_eval_max_batch_count,
                 q_and_a_patience, q_and_a_eps, q_and_a_max_batch_count,
                 current_module="cls",
                 verbose=False):
        self.net = net

        self.modules_patience = {
            'cls' : cls_patience,
            'conf' : conf_eval_patience,
            'q_a' : q_and_a_patience,
        }
        self.current_patience = 0

        self.modules_eps = {
            'cls': cls_eps,
            'conf': conf_eval_eps,
            'q_a': q_and_a_eps
        }
        self.modules_max_epoch = {
            'cls' : cls_max_batch_count,
            'conf' : conf_eval_max_batch_count,
            'q_a' : q_and_a_max_batch_count
        }

        self.modules_history = {
            'cls' : [],
            'conf':[],
            'q_a':[]
        }

        self.net_params = {
            'cls' : [self.net.base_module,self.net.f_cls,],
            'conf' : [self.net.conf_eval,],
            'q_a' : [self.net.q_m,self.net.a_m,],
        }

        self.current_module = current_module
        self.verbose = verbose

    def step(self, modules_losses):
        self._set_requires_grad()

        loss = modules_losses[self.current_module].mean()
        loss.backward(retain_graph = True)
        loss = loss.item()
        if len(self.modules_history[self.current_module]) >= self.modules_max_epoch[self.current_module]:
            self._change_module()
        else:
            if len(self.modules_history[self.current_module]) > 0:
                if loss > min(self.modules_history[self.current_module][-self.modules_patience[self.current_module]:]) - self.modules_eps[self.current_module]:
                    self.current_patience += 1
                    if self.current_patience >= self.modules_patience[self.current_module]:
                        self._change_module()
                else:
                    self.current_patience = 0

            self.modules_history[self.current_module].append(loss)

        return loss

    def _change_module(self):
        next_module = {
            'cls': 'conf',
            'conf': 'q_a',
            'q_a': 'cls',
        }

        self.current_module = next_module[self.current_module]
        self.modules_history[self.current_module] = []
        self.current_patience = 0

        if self.verbose:
            print("Changing module to ",self.current_module)

    def _set_requires_grad(self):
        for param in self.net.parameters():
            param.requires_grad = False
        for module in self.net_params[self.current_module]:
            for param in module.parameters():
                param.requires_grad = True

