"""
加载训练集、验证集以及测试集
"""
import os, json, pickle
import numpy as np


# 根据sc_type获取不同智能合约漏洞数据集的地址
def get_data_path_vocab(sc_type):
    if sc_type == "RE":
        train_data_path = "RE_Paths/re_code_path_training.json"
        valid_data_path = "RE_Paths/re_code_path_valid.json"
        test_data_path = "RE_Paths/re_code_path_test.json"
        vocab2id_path = "RE_Paths/RE_vocab_id.pkl"
    elif sc_type == "TD":
        train_data_path = "TD_Paths/re_code_path_training.json"
        valid_data_path = "TD_Paths/re_code_path_valid.json"
        test_data_path = "TD_Paths/re_code_path_test.json"
        vocab2id_path = "RE_Paths/TD_vocab_id.pkl"
    elif sc_type == "OF":
        train_data_path = "OF_Paths/re_code_path_training.json"
        valid_data_path = "OF_Paths/re_code_path_valid.json"
        test_data_path = "OF_Paths/re_code_path_test.json"
        vocab2id_path = "RE_Paths/OF_vocab_id.pkl"
    elif sc_type == "DE":
        train_data_path = "DE_Paths/re_code_path_training.json"
        valid_data_path = "DE_Paths/re_code_path_valid.json"
        test_data_path = "DE_Paths/re_code_path_test.json"
        vocab2id_path = "RE_Paths/DE_vocab_id.pkl"
    return train_data_path, valid_data_path, test_data_path, vocab2id_path



# 从SC数据集加载数据；每一条json数据中的键值为：sol_name, label, first path, second path, third path, long path
# 由于要将4条路径进行拼接，所以使用max_setence_length进行截断或补全，来确保路径长度一致
def get_SC_data(sc_json_path, max_setence_length):
    # 初始化合约数据及标签的列表
    sc_data, sc_label = [], []

    # 读取所有数据
    with open(sc_json_path, 'r', encoding='utf-8') as file:
        # 使用json.load()方法解析JSON数据
        all_data = file.readlines()

    for i in all_data:
        if i.strip() is "":
            continue
        one_sol_json = json.loads(i)
        first_path_list = one_sol_json["first path"].split(" ")
        # 计算first_path_list的长度，判断是否需要进行截断
        if len(first_path_list) < max_setence_length:
            first_path_list.extend(["PAD"]*(max_setence_length-len(first_path_list)))
        else:
            first_path_list = first_path_list[:max_setence_length]

        second_path_list = one_sol_json["second path"].split(" ")
        # 计算second_path_list的长度，判断是否需要进行截断
        if len(second_path_list) < max_setence_length:
            second_path_list.extend(["PAD"] * (max_setence_length - len(second_path_list)))
        else:
            second_path_list = second_path_list[:max_setence_length]

        third_path_list = one_sol_json["third path"].split(" ")
        # 计算third_path_list的长度，判断是否需要进行截断
        if len(third_path_list) < max_setence_length:
            third_path_list.extend(["PAD"] * (max_setence_length - len(third_path_list)))
        else:
            third_path_list = third_path_list[:max_setence_length]

        long_path_list = one_sol_json["long path"].split(" ")
        # 计算first_path_list的长度，判断是否需要进行截断
        if len(long_path_list) < max_setence_length:
            long_path_list.extend(["PAD"] * (max_setence_length - len(long_path_list)))
        else:
            long_path_list = long_path_list[:max_setence_length]

        temp_path = first_path_list + second_path_list + third_path_list + long_path_list
        sc_data.append(temp_path)
        sc_label.append(one_sol_json["label"])

    # print("The length of train data is {0}, and max_setence_length is {1}".format(len(op_data), max_setence_length))
    return np.array(sc_data), np.array(sc_label)


# 加载本地训练数据
def load_train_valid_test_data(SC_Type, max_setence_length):

    train_data_path, valid_data_path, test_data_path, vocab2id_path = get_data_path_vocab(SC_Type)

    # 加载词汇表
    with open(vocab2id_path, 'rb') as f:
        vocab2id = pickle.load(f)

    # 加载训练数据，并且ont-hot向量化，对应word2vec词嵌入
    temp_data, train_label = get_SC_data(train_data_path, max_setence_length)
    train_data = []
    for i in temp_data:
        temp = []
        for j in i:
            if j in vocab2id:
                temp.append(vocab2id[j])
            else:
                temp.append(vocab2id["PAD"])
        train_data.append(temp)

    # 加载验证数据，并且ont-hot向量化，对应word2vec词嵌入
    temp_data, valid_label = get_SC_data(valid_data_path, max_setence_length)
    valid_data = []
    for i in temp_data:
        temp = []
        for j in i:
            if j in vocab2id:
                temp.append(vocab2id[j])
            else:
                temp.append(vocab2id["PAD"])
        valid_data.append(temp)

    # 加载测试数据，并且ont-hot向量化，对应word2vec词嵌入
    temp_data, test_label = get_SC_data(test_data_path, max_setence_length)
    test_data = []
    for i in temp_data:
        temp = []
        for j in i:
            if j in vocab2id:
                temp.append(vocab2id[j])
            else:
                temp.append(vocab2id["PAD"])
        test_data.append(temp)

    return np.array(train_data), train_label, np.array(valid_data), valid_label, np.array(test_data), test_label