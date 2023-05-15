from flask import Flask, request, jsonify
import json
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# 加载模型
print("正在加载模型，请稍等...")
model = RWKV(model='/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16')
pipeline = PIPELINE(model, "20B_tokenizer.json")
print("模型加载完成！")

# 设置参数
args = PIPELINE_ARGS(temperature = 1.0, max_tokens = 50, top_p = 0.95, frequency_penalty = 0.0, presence_penalty = 0.0, stop=["\n"])

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
    out = pipeline(prompt, **args)
    # 将模型的输出写入该usrid下的记录列表，并在末尾添加一个换行符
    chat_dict[usrid].append(out + "\n")
    # 将该usrid下的所有记录拼接起来，作为响应返回
    response = ''.join(chat_dict[usrid])
    # 输出响应内容
    print(f"响应内容：{response}")
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)