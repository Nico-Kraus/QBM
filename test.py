# import math
# x = 36
# sigx = 1 / (1 + math.exp(-x))
# print(sigx)
# log = math.log(sigx)
# print(log)
# log2 = math.log(1-sigx)
# print(log2)

def append_list(item, list=[]):
    list.append(item)
    return list

x = append_list(1)
y = append_list(2)
z = append_list(3)
print(x)