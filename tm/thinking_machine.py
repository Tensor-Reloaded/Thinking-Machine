import torch
import torch.nn as nn

import torch.nn.functional as F

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size= 1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)

        return x


class FinalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*9*9, 10)
        self.sm = nn.Softmax(-1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        # return self.sm(x)
        return x


class ConfidenceEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 64, 1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return torch.sigmoid(x)

class QueryMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 16, 1)
        # self.max_pool = nn.MaxPool2d(kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        # x = self.max_pool(x)
        return x

class AnsMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32 + 16, 32, kernel_size=1)

    def forward(self, x, q):
        unified = torch.cat((q, x), dim=1)

        return torch.tanh(self.conv1(unified))




class TM(nn.Module):
    def __init__(self, conf_threshold = 0.9, max_depth = 8):
        super().__init__()
        self.base_module = BaseModule()
        self.max_depth = max_depth
        self.conf_threshold = conf_threshold
        self.f_cls = FinalClassifier()
        self.conf_eval = ConfidenceEval()
        self.q_m = QueryMachine()
        self.a_m = AnsMachine()

    def forward(self, mini_batch):
        #todo different forward for train and eval

        mini_batch = self.base_module(mini_batch)
        outputs = []
        all_conf = []
        all_a = []
        all_q = []
        all_f_cls = []

        for x in mini_batch:
            x = x.unsqueeze(0)
            depth = 0
            current_confs = []
            current_as = []
            current_qs = []
            current_classes = []
            while depth < self.max_depth:
                current_conf = self.conf_eval(x)  
                if self.training: # We only have to log intermediary classifications when training 
                    current_confs.append(current_conf)
                    current_classes.append(self.f_cls(x))

                if current_conf >= self.conf_threshold:
                    break

                q = self.q_m(x)
                a = self.a_m(x, q)

                if self.training: 
                    current_qs.append(q)
                    current_as.append(a)

                x = x + a
                depth += 1

            final_class = self.f_cls(x) # TODO: refactor to use the last classification from the while loop aka. 1 less f_cls(x) call, probably done with an "else" at the "while" loop above

            outputs.append(final_class)

            if not self.training:
                continue
            
            current_classes.append(final_class)

            all_conf.append(torch.cat(current_confs))

            if depth and self.training: # TODO: Refactor the appends in a cleaner version based on self.training
                all_q.append(torch.cat(current_qs))
                all_a.append(torch.cat(current_as))
                all_f_cls.append(torch.cat(current_classes))
            else:
                all_q.append([])
                all_a.append([])
                all_f_cls.append(torch.cat(current_classes))

        if not self.training:
            return torch.cat(outputs)

        return  torch.cat(outputs), \
                all_conf, \
                all_q, \
                all_a, \
                all_f_cls








