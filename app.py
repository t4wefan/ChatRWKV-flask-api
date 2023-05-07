print("starting chatrwkv server")

import os
import time
print("finding CUDA")
import torch
from torch.utils.cpp_extension import CUDA_HOME
from flask import Flask, jsonify, request
from rwkvstic.load import RWKV

app = Flask(__name__)

# 检测 CUDA 是否可用并输出 CUDA 设备名称和 CUDA 安装路径
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print("CUDA device found:", device_name)

    cuda_home = CUDA_HOME
    if cuda_home is None or cuda_home.strip() == '':
        print("CUDA_HOME is empty, please check your CUDA driver")
    else:
        os.environ['CUDA_HOME'] = cuda_home
        print("CUDA home:", cuda_home)
else:
    print("CUDA device not found")

# 输出 "loading model"，加载模型并输出 "model loaded"
print("loading model")
model = RWKV(
    "https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-Instruct-test1-20230124.pth"
)
print("model loaded")

# 修改 /chatrwkv 路由，同时支持 GET 和 POST 请求
@app.route('/chatrwkv', methods=['GET', 'POST'])
def chat_with_rwkv():

    # 如果是 GET 请求
    if request.method == 'GET':
        # 从请求参数中获取 msg、usrid 和 source
        msg = request.args.get('msg')
        usrid = request.args.get('usrid')
        source = request.args.get('source')
    # 如果是 POST 请求
    elif request.method == 'POST':
        # 从请求参数中获取 msg、usrid 和 source
        msg = request.form.get('msg')
        usrid = request.form.get('usrid')
        source = request.form.get('source')
    else:
        # 如果不是 GET 或 POST 请求，则返回错误响应
        return jsonify({'status': 'error', 'error': 'method not allowed'}), 405

    # 如果 usrid 参数不存在或为空，则返回错误响应
    if not usrid or usrid.strip() == '':
        return jsonify({'status': 'error', 'error': 'usrid parameter is missing or empty'}), 400

    # 如果 source 参数不存在或为空，则返回错误响应
    if not source or source.strip() == '':
        return jsonify({'status': 'error', 'error': 'source parameter is missing or empty'}), 400

    # 如果 msg 参数不存在或为空，则返回错误响应
    if not msg or msg.strip() == '':
        return jsonify({'status': 'error', 'error': 'msg parameter is missing or empty'}), 400

    # 构建聊天历史记录文件名和路径
    filename = f"{usrid}.txt"
    filepath = os.path.join(os.path.dirname(__file__), 'history', filename)

    # 如果聊天历史记录文件不存在，则创建文件
    if not os.path.exists(filepath):
        open(filepath, 'w').close()

    # 将消息内容写入聊天历史记录文件
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")

    # 调用 RWKV 模型进行聊天
    res = model.predict(msg)

    # 将聊天结果写入聊天历史记录文件
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"{res}\n")

    # 将聊天结果写入响应中并返回
    return jsonify({'status': 'ok', 'reply': res})

# 启动 Flask 应用程序
app.run(host='0.0.0.0', port=7860)