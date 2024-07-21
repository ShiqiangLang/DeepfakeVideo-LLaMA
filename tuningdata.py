import os
import json

# 设置输入文件夹路径
input_folder = "/home/lsq/FF++/manipulated_sequences/NeuralTextures/c23/videos/"

# 尝试读取之前生成的JSON文件
try:
    with open("output.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    # 如果文件不存在,则创建一个空列表
    data = []

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
        # 创建一个字典,存储视频和问答信息
        video_data = {
            "video": os.path.join(input_folder, filename),
            "QA": [
                {
                    "q": "Tell me if there are synthesis artifacts in the face or not. Must return with yes or no only.",
                    "a": "yes."
                }
            ]
        }
        # 将视频和问答信息添加到数据列表中
        data.append(video_data)

# 将数据列表转换为JSON格式并写入文件
with open("output.json", "w") as f:
    json.dump(data, f, indent=2)

print("JSON file created successfully!")