import os
import torch
import torch.utils.cpp_extension
from flask import Flask, jsonify, request
from rwkvstic.load import RWKV

# 加载 RWKV 模型
model = RWKV(
    "https://huggingface.co/BlinkDL/rwkv-4-pile-3b/resolve/main/RWKV-4-Pile-3B-Instruct-test1-20230124.pth"
)

app = Flask(__name__)

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
    filepath = os.path.join(os.getcwd(), 'history', filename)

    # 如果聊天历史记录文件不存在，则创建一个空文件
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('')

    # 将 msg 写入聊天历史记录文件
    with open(filepath, 'a') as f:
        f.write(f"Ask: {msg}\n")

    # 获取聊天历史记录文件中对应 usrid 的全部内容
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith(f"{usrid}:"):
                context = "".join(lines[i:]).strip()
                break

    # 设置模型上下文
    model.loadContext(newctx=context)

    # 生成回答
    output = model.forward(number=1)["output"][0]

    # 从回答中提取最后一行 reply，并去掉开头的 "Reply: " 字符串
    reply = output.split("Reply:")[-1].strip().replace("Reply:", "").strip()

    # 将 reply 写入聊天历史记录文件
    with open(filepath, 'a') as f:
        f.write(f"Reply: {reply}\n\n")

    # 返回响应
    response = {
        'status': 'ok',
        'usrid': usrid,
        'reply': reply
    }
    return jsonify(response), 200

if __name__ == '__main__':
    # 获取 CUDA 的安装路径
    cuda_home = torch.utils.cpp_extension.CUDA_HOME

    # 将 CUDA 的安装路径写入环境变量
    os.environ['CUDA_HOME'] = cuda_home

    # 启动 Flask 应用程序
    app.run(host='0.0.0.0', port=7860)