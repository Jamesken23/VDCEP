import torch, os
import torch.utils.data as Data
from imblearn.over_sampling import RandomOverSampler, SMOTE

from Network import vdcep_model
from utils.log_helper import get_logger, get_log_path
from Datasets.load_data import load_train_valid_test_data

from utils.config import create_parser
import Network.vdcep_trainer as train_sl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = create_parser()


def get_balanced_data(inputs, labels):
    # ros = RandomOverSampler(random_state=39)
    # inputs, labels = ros.fit_resample(inputs, labels.ravel())

    sm = SMOTE(random_state=39)
    inputs, labels = sm.fit_resample(inputs, labels.ravel())

    return inputs, labels


def create_loader(args, log_set):
    lab_batch_size, test_batch_size = args.batch_size, args.batch_size
    SC_Type, max_setence_length = args.SC_Type, args.max_setence_length
    train_data, train_label, valid_data, valid_label, test_data, test_label = load_train_valid_test_data(SC_Type, max_setence_length)

    log_set.info("Original shape information: labeled train data is {0}, valid data is {1}, test data is {2}".format(
        train_data.shape, valid_data.shape, test_data.shape))

    if args.is_balanced:
        train_data, train_label = get_balanced_data(train_data, train_label)
        valid_data, valid_label = get_balanced_data(valid_data, valid_label)
        test_data, test_label = get_balanced_data(test_data, test_label)

    log_set.info("Original shape information: labeled train data is {0}, valid data is {1}, test data is {2}".format(
        train_data.shape, valid_data.shape, test_data.shape))

    train_inputs, train_labels = torch.LongTensor(train_data), torch.LongTensor(train_label)
    valid_inputs, valid_labels = torch.LongTensor(valid_data), torch.LongTensor(valid_label)
    test_inputs, test_labels = torch.LongTensor(test_data), torch.LongTensor(test_label)


    # 加载训练集、验证集、测试集的张量形式
    train_dataset = Data.TensorDataset(train_inputs, train_labels)
    valid_dataset = Data.TensorDataset(valid_inputs, valid_labels)
    test_dataset = Data.TensorDataset(test_inputs, test_labels)

    train_loader = Data.DataLoader(train_dataset, batch_size=lab_batch_size, shuffle=True, num_workers=2)
    val_loader = Data.DataLoader(valid_dataset, batch_size=test_batch_size, num_workers=2)
    test_loader = Data.DataLoader(test_dataset, batch_size=len(test_data), num_workers=2)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 一些关键设置
    args.SC_Type, args.batch_size = "RE", 32
    args.epochs = 100

    # 开始打印日志信息
    log_path, log_name = get_log_path(args)
    log_set = get_logger(log_path)

    # 获取embedding model
    emb_model = vdcep_model.vdcep(vocab_size=args.vocab_size)

    log_set.info(
        "We select {0} embedding model, and use {1} smart contract vulnerability. Our lab_batch_size is {2}".format(
            args.model, args.SC_Type, args.batch_size))

    train_loader, val_loader, test_loader = create_loader(args, log_set)
    sc_train = train_sl.Trainer(args.max_setence_length, emb_model, log_set, device)

    sc_train.loop(args.epochs, train_loader, val_loader, test_loader)