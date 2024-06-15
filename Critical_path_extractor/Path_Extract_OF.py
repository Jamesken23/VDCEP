"""
对于整数溢出/下溢漏洞，给定一条代码执行路径，判断其是否是critical code execution path
"""
import re, os



# 读取合约
def Read_Sol(sol_path):
    with open(sol_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    return all_lines


def Is_Critical_Path(sol_content):
    priority_cnt = 0

    # 模式1：判断是否直接包含msg.sender的关键词
    for one_line in sol_content:
        if 'msg.sender' in one_line:
            priority_cnt += 1


    # 模式2：判断是否存在包含msg.sender的函数
    function_flag = 0
    for one_line in sol_content:
        if 'msg.sender' in one_line:
            function_flag += 1
        elif function_flag > 0:
            one_line = one_line.replace(" ", "")
            if "-" in one_line or "-=" in one_line or "=0" in one_line:
                priority_cnt += 1


    # 模式3：判断是否存在和 user balance 有关的代码语句
    function_flag, param = 0, None
    for one_line in sol_content:
        if '.call.value' in one_line:
            param = re.findall(r".call.value\((.+?)\)", one_line)
            if len(param) > 0:
                param = param[0]

            function_flag += 1
        elif function_flag > 0:
            if param in one_line:
                priority_cnt += 1


    if priority_cnt > 0:
        is_critical = True
    else:
        is_critical = False
    return is_critical, priority_cnt



if __name__ == "__main__":
    sol_path = r"simple_re_sol.txt"
    sol_content = Read_Sol(sol_path)
    print("sol_content: ", sol_content)
    is_critical, priority_cnt = Is_Critical_Path(sol_content)
    if is_critical:
        print("{0} is critical, and the priority_cnt is {1}.".format(sol_path, priority_cnt))
    else:
        print("{0} is not critical, and the priority_cnt is {1}.".format(sol_path, priority_cnt))