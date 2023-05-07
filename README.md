# ChatRWKV-flask-api
一个由 ChatGPT 完成的 ChatRWKV 服务端，包括这个   readme（的以下部分

# 程序功能
本程序是一个简单的基于 Flask 框架的聊天机器人服务端，使用 RWKV 模型进行聊天，支持 GET 和 POST 请求，用于与客户端交互并记录聊天记录。

# 推荐安装要求
本程序推荐使用 Python 3.10 或以上版本以及 Flask 和 torch 库，并需要安装 rwkvstic 库才能正常运行。

# 安装依赖
可以直接通过以下命令安装依赖：
```
pip install -r requirements.txt
```

# API调用方法
1. 启动程序后，通过 GET 或 POST 请求访问 /chatrwkv 路由。
2. 在请求参数中传递消息内容、用户 ID 和消息来源。
3. 服务器将会返回聊天机器人的回复，并将聊天记录保存到文件中。

# 调用参数
/chatrwkv 路由支持以下请求参数：

- `msg`：要发送给聊天机器人的消息内容。
- `usrid`：用户 ID。
- `source`：消息来源。

注意：以上三个参数都是必须的，否则服务器将会返回错误响应。