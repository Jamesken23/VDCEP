import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(self, max_setence_length, model, log_set, device):
        self.max_set_len = max_setence_length
        self.log_set = log_set
        self.optimizer = model.optimizer
        self.device = device
        self.model = model.to(self.device)
        self.best_model = None

        self.ce_loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)

    def train_iteration(self, train_x_loader):

        label_acc, all_loss = 0., 0.
        num_step = 0
        for label_x, label_y in train_x_loader:
            num_step += 1
            label_x, label_y = label_x.to(self.device), label_y.to(self.device)

            # === decode targets of unlabeled data ===
            lbs = label_x.size(0)
            # 对输入数据进行切片
            c_path_1, c_path_2 = label_x[:, :self.max_set_len], label_x[:, self.max_set_len: self.max_set_len*2]
            c_path_3, l_path = label_x[:, self.max_set_len*2: self.max_set_len*3], label_x[:, self.max_set_len*3: self.max_set_len*4]

            # === forward ===
            outputs = self.model(c_path_1, c_path_2, c_path_3, l_path)
            supervised_loss = self.ce_loss(outputs, label_y)

            all_loss += supervised_loss.item()

            ## backward
            self.optimizer.zero_grad()
            supervised_loss.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = label_y.eq(outputs.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs

        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and all training loss is {1}".format(
            label_acc / float(num_step), all_loss / float(num_step)))

    def train(self, label_loader):
        self.model.train()

        with torch.enable_grad():
            self.train_iteration(label_loader)

    def val_iteration(self, data_loader):
        acc, num_step = 0., 0
        for _, (data, targets) in enumerate(data_loader):
            num_step += 1
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            # 对输入数据进行切片
            c_path_1, c_path_2 = data[:, :self.max_set_len], data[:, self.max_set_len: self.max_set_len * 2]
            c_path_3, l_path = data[:, self.max_set_len * 2: self.max_set_len * 3], data[:,
                                                                                       self.max_set_len * 3: self.max_set_len * 4]
            outputs = self.model(c_path_1, c_path_2, c_path_3, l_path)

            test_acc = targets.eq(outputs.max(1)[1]).float().sum().item()
            acc += test_acc / data.size(0)

        self.log_set.info(">>>>>[test] test data's accuracy is {0}".format(acc / float(num_step)))
        return acc / float(num_step)

    def validate(self, data_loader):
        self.model.eval()

        with torch.no_grad():
            return self.val_iteration(data_loader)

    def predict(self, model, data_loader):
        model.eval()
        pred_list, y_list = [], []
        for _, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            # 对输入数据进行切片
            c_path_1, c_path_2 = data[:, :self.max_set_len], data[:, self.max_set_len: self.max_set_len * 2]
            c_path_3, l_path = data[:, self.max_set_len * 2: self.max_set_len * 3], data[:,
                                                                                    self.max_set_len * 3: self.max_set_len * 4]
            outputs = self.model(c_path_1, c_path_2, c_path_3, l_path)

            if torch.cuda.is_available():
                y_label = targets.cpu().detach().numpy().tolist()
                pred = outputs.cpu().detach().numpy().tolist()
            else:
                y_label = targets.detach().numpy().tolist()
                pred = outputs.detach().numpy().tolist()

            pred_list.extend(pred)
            y_list.extend(y_label)

        # print("pred_list shape is {0}, and y_list shape is {1}".format(np.array(pred_list).shape, np.array(y_list).shape))
        tn, fp, fn, tp = confusion_matrix(y_list, np.argmax(pred_list, axis=1)).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)

        recall, precision = tp / (tp + fn + 0.000001), tp / (tp + fp + 0.000001)
        F1 = (2 * precision * recall) / (precision + recall + 0.000001)

        return acc, recall, precision, F1

    # 主函数
    def loop(self, epochs, train_data, val_data, test_data):

        best_acc, best_epoch = 0., 0
        for ep in range(epochs):
            self.epoch = ep
            self.log_set.info("---------------------------- Epochs: {} ----------------------------".format(ep))
            self.train(train_data)

            val_acc = self.validate(val_data)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = ep
                self.best_model = deepcopy(self.model).to(self.device)

        acc, recall, precision, F1 = self.predict(self.model, test_data)
        self.log_set.info(
            "Final epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(epochs, acc,
                                                                                                            recall,
                                                                                                            precision,
                                                                                                            F1))
        acc, recall, precision, F1 = self.predict(self.best_model, test_data)
        self.log_set.info(
            "The best epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(
                best_epoch, acc, recall, precision, F1))