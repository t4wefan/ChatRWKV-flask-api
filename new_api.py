print('初始化中')
from flask import Flask, request, jsonify
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import os, sys, torch
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# 加载模型
print("正在加载模型，请稍等...")
filename = 'RWKV-4-Raven-3B-v10-Eng49%-Chn50%-Other1%-20230419-ctx4096.pth'
def checkmodel(filename):
    if os.path.isfile(filename):
        print('模型以存在，开始加载')
    else:
        print('模型不存在，开始下载')
        os.system('wget https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-3B-v10-Eng49%25-Chn50%25-Other1%25-20230419-ctx4096.pth')
        return 'done'

checkmodel(filename)

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
model = RWKV(model='RWKV-4-Raven-3B-v10-Eng49%-Chn50%-Other1%-20230419-ctx4096', strategy='cuda fp16')
pipeline = PIPELINE(model, "20B_tokenizer.json")

#out, state = model.forward([187, 510, 1563, 310, 247], None)
#print(out.detach().cpu().numpy())                   # get logits
#out, state = model.forward([187, 510], None)
#out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
#out, state = model.forward([310, 247], state)
#print(out.detach().cpu().numpy())

print("模型加载完成！")

# 创建Flask应用
app = Flask(__name__)

# 创建空字典，用于存储对话记录
chat_dict = {}

# 定义路由函数
@app.route('/chatrwkv', methods=['GET'])
def chat_rwkv():
    # 获取请求参数
    source = request.args.get('source')
    msg = request.args.get('msg')
    usrid = request.args.get('usrid')
    # 检查参数是否齐全
    if not all([source, msg, usrid]):
        return jsonify({'code': 400, 'msg': '参数缺失'})
    # 输出请求参数
    print(f"请求参数：source={source}, msg={msg}, usrid={usrid}")
    # 如果该usrid还没有对话记录，就创建一个空列表，并将其作为chat_dict的一个键值对，键为usrid，值为该列表
    if usrid not in chat_dict:
        chat_dict[usrid] = []
    # 将msg参数写入该usrid下的记录列表，并在末尾添加一个换行符
    chat_dict[usrid].append(msg + "\n")
    # 将该usrid下的所有记录拼接起来，作为输入给模型，并调用rwkv模型生成回答
    prompt = ''.join(chat_dict[usrid])
    ctx = prompt
    out = "out"
    args = PIPELINE_ARGS(temperature=max(0.2, float(0.99)), top_p=float(0.99),
                         token_ban=[],  # ban the generation of some tokens
                         token_stop=[0])  # stop generation whenever you see any token here
    out = pipeline.generate(ctx,  )
    # 将模型的输出写入该usrid下的记录列表，并在末尾添加一个换行符
    chat_dict[usrid].append(out + "\n")
    # 将该usrid下的所有记录拼接起来，作为响应返回
    response = out
    # 输出响应内容
    print(f"响应内容：{response}")
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)