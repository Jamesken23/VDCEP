import os.path
import re, json

# 合约文件名的正则化匹配模式
sol_pattern = r'(.*)\.sol'

read_sol_cnt = 0

# 将数据集中的数据保存至Json文件中
def get_path_json(ori_vul_data_path, code_execution_paths, can_read_sol=False):
    # 初始化一个合约文件名及列表
    sol_name, sol_content_list, sol_name_dict = "1.sol", [], {}

    # 读取源文件中的合约数据
    with open(ori_vul_data_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        for i in range(len(all_lines)):
            # 读取每一行代码
            one_line_code = all_lines[i].strip()
            if one_line_code != "---------------------------------":
                if can_read_sol is False:
                    sol_result = re.search(sol_pattern, one_line_code)
                    next_sol_result = re.search(sol_pattern, all_lines[i+1].strip())
                    # 如果连续两行都包括"sol"文件，退出，否则，可以记录当前文件名
                    if sol_result and next_sol_result:
                        continue
                    else:
                        sol_name = one_line_code
                        can_read_sol = True
                else:
                    sol_content_list.append(one_line_code)
            else:
                if len(sol_content_list) > 0:

                    sol_label = int(all_lines[i - 1].strip())
                    # 将合约的标签吐出来
                    sol_content_list.pop()

                    sol_str = ""
                    for i in sol_content_list:
                        sol_str += i + " "

                    sol_content_list.clear()

                    one_sol_dict = {}
                    one_sol_dict["sol name"] = sol_name
                    one_sol_dict["label"] = sol_label

                    one_sol_dict["first path"] = sol_str
                    one_sol_dict["second path"] = sol_str
                    one_sol_dict["third path"] = sol_str

                    if sol_name not in sol_name_dict:
                        sol_name_dict[sol_name] = 1
                    else:
                        sol_name_dict[sol_name] += 1
                        can_read_sol = False
                        continue

                    #  保存数据,每个合约数据以换行结束
                    with open(code_execution_paths, 'a') as ff:
                        json.dump(one_sol_dict, ff)
                        ff.write('\n')
                    print("We have read {0}".format(sol_name))

                can_read_sol = False

    print("We have converted {0} sol".format(len(sol_name_dict)))


if __name__ == '__main__':
    # ReChecker工作的数据集位置（含文件夹）
    ori_vul_data_path = r"D:\Python\Python_Projects\VDCEP\Datasets\reentrancy\graph_data_fragment_test_200.txt"
    # 读取数据（对应代码执行路径）的保存位置
    code_execution_paths = r"D:\Python\Python_Projects\VDCEP\Datasets\RE_Paths\re_code_paths_test.json"

    # 开始生成路径
    get_path_json(ori_vul_data_path, code_execution_paths, can_read_sol=False)







# with open(ori_vul_data_path, 'r', encoding='utf-8') as f:
#     all_lines = f.readlines()
#     for i in range(len(all_lines)):
#         one_line = all_lines[i].strip()
#         result = re.search(pattern, one_line)
#         if result:
#             # 提取匹配结果中的前缀部分
#             prefix = result.group(1)
#             # 输出前缀部分
#             print(prefix)

