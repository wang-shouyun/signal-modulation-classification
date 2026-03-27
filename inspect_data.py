import pickle
import os

data_path = os.path.join("data", "RML2016.10a_dict.pkl")

print("数据文件路径：", data_path)
print("文件是否存在：", os.path.exists(data_path))

with open(data_path, "rb") as f:
    Xd = pickle.load(f, encoding="latin1")

print("数据类型：", type(Xd))
print("键的数量：", len(Xd))

keys = list(Xd.keys())
print("前5个键：", keys[:5])

first_key = keys[0]
print("第一个键：", first_key)
print("对应数据类型：", type(Xd[first_key]))
print("对应数据形状：", Xd[first_key].shape)