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
    def __init__(self, device, conf_threshold = 0.95, max_depth= 5):
        super().__init__()
        self.device = device
        self.base_module = BaseModule()
        self.max_depth = max_depth
        self.conf_threshold = conf_threshold
        self.f_cls = FinalClassifier()
        self.conf_eval = ConfidenceEval()
        self.q_m = QueryMachine()
        self.a_m = AnsMachine()

    def forward(self, mini_batch):
        #todo different forward for train and eval

        all_confs = []
        all_f_cls = []

        batch_size = len(mini_batch)

        actual_depth = torch.ones(batch_size, device=self.device, dtype=torch.long) * self.max_depth

        x = self.base_module(mini_batch)

        depth = 0

        while depth < self.max_depth and actual_depth.max() == self.max_depth:
            depth += 1
            current_confs = self.conf_eval(x)
            current_f_cls = self.f_cls(x)
            acceptable_conf = (( current_confs.squeeze() > self.conf_threshold).nonzero()).view(-1)

            # print(acceptable_conf)

            depth_tensor = torch.ones(len(acceptable_conf), device= self.device, dtype=torch.long) * depth
            actual_depth[acceptable_conf] = torch.min(input = actual_depth[acceptable_conf], other = depth_tensor)

            all_confs.append(current_confs)
            all_f_cls.append(current_f_cls)

            q = self.q_m(x)
            a = self.a_m(x, q)

            x = x + a

        only_valid_f_cls = []
        only_valid_confs = []
        outputs = []

        all_f_cls = torch.stack(all_f_cls).transpose(0, 1)
        all_confs = torch.stack(all_confs).transpose(0, 1).squeeze(-1)

        for sample_idx in range(batch_size):
            if actual_depth[sample_idx]==0:
                print(actual_depth[sample_idx])
                #exit()
            only_valid_f_cls.append(all_f_cls[sample_idx, : actual_depth[sample_idx]])
            only_valid_confs.append(all_confs[sample_idx, : actual_depth[sample_idx]])

            outputs.append(all_f_cls[sample_idx, actual_depth[sample_idx] - 1, :])
        outputs = torch.stack(outputs)

        #return all_f_cls, all_confs, actual_depth

        if self.training:
            return outputs, only_valid_f_cls, only_valid_confs, actual_depth
        else:
            return outputs


