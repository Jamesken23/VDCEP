import os
import pickle, json


# 读取某个漏洞数据集下的所有数据
def get_all_data(sc_data_path):
    all_sol = []
    # 首先读取三条代码执行路径
    with open(sc_data_path, 'r', encoding='utf-8') as file:
        # 使用json.load()方法解析JSON数据
        all_data = file.readlines()

    for i in all_data:
        if i.strip() is "":
            continue
        one_sol_json = json.loads(i)
        long_path_list = one_sol_json["long path"].split(" ")
        all_sol.extend(long_path_list)

    return all_sol



# 获取词汇表，用于one-hot向量
def word_process(sc_data_path, sc_type):
    word_list = []

    all_sol_data = get_all_data(sc_data_path)
    # 清除一些噪声字符
    for i in all_sol_data:
        if i in ['{', '}', '(', ')', ' ', ';', ',']:
            continue
        if '0x' in i:
            continue
        word_list.append(i.strip())

    word_set = list(set(word_list))
    vocab2id = {w: i + 1 for i, w in enumerate(word_set)}
    vocab2id["PAD"] = 0

    # 将某个漏洞数据集下的词汇表保存
    vocab_dir = sc_data_path.split("/")[0]
    vocab_path = os.path.join(vocab_dir,  sc_type + "_vocab_id.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab2id, f)

    print("We have got vocab file! This file contains {0} vocab".format(len(vocab2id)))


if __name__ == "__main__":
    sc_data_path = r"RE_Paths/all_re_code_paths.json"
    sc_type = "RE"

    word_process(sc_data_path, sc_type)