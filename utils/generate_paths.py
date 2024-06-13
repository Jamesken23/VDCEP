"""
生成VDCEP所需的4条代码执行路径，对应Step 1: 数据处理
"""
import re, json


# 合约文件名的正则化匹配模式
sol_pattern = r'(.*)\.sol'


# 读取所有合约的原始数据
def generate_all_sol(ori_vul_data_path, can_read_sol=False):
    all_sol_dict = {}
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
                    # 如果连续两行都包括"sol"文件，退出，否则，可以记录当前文件名
                    if sol_result:
                        sol_name = one_line_code.split(" ")[-1]
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
                    all_sol_dict[sol_name] = sol_str
                can_read_sol = False
    print("We have converted {0} sol".format(len(all_sol_dict)))
    return all_sol_dict


# 生成合约的最长路径
def generate_long_path(all_sol_dict, code_execution_paths, new_code_execution_paths):

    # 首先读取三条代码执行路径
    with open(code_execution_paths, 'r', encoding='utf-8') as file:
        # 使用json.load()方法解析JSON数据
        all_data = file.readlines()

    for i in all_data:
        one_sol_json = json.loads(i)
        one_sol_json["long path"] = all_sol_dict[one_sol_json["sol name"]]

        #  保存数据,每个合约数据以换行结束
        with open(new_code_execution_paths, 'a') as ff:
            json.dump(one_sol_json, ff)
            ff.write('\n')



if __name__ == '__main__':
    # ReChecker工作的数据集位置（含文件夹）
    ori_vul_data_path = r"D:\Python\Python_Projects\VDCEP\Datasets\reentrancy\graph_data_smart_contract_2000.txt"
    # 读取数据（对应代码执行路径）的保存位置
    code_execution_paths = r"D:\Python\Python_Projects\VDCEP\Datasets\RE_Paths\re_code_paths_test.json"
    # 读取数据（对应代码执行路径）的保存位置
    all_sol_dict = generate_all_sol(ori_vul_data_path, can_read_sol=False)

    # 4条路径最终保存的位置
    new_code_execution_paths = r"D:\Python\Python_Projects\VDCEP\Datasets\RE_Paths\re_all_code_paths_test.json"
    generate_long_path(all_sol_dict, code_execution_paths, new_code_execution_paths)