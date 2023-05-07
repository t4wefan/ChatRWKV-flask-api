from flask import Flask, jsonify, request
import os

app = Flask(__name__)

@app.route('/chat/history', methods=['GET', 'POST'])
def get_chat_history():
    usrid = request.args.get('usrid') or request.form.get('usrid')
    content = request.form.get('content')

    if not usrid:
        return jsonify({'error': 'usrid parameter is missing'}), 400

    if request.method == 'POST' and not content:
        return jsonify({'error': 'content parameter is missing'}), 400

    filename = f"{usrid}.txt"
    filepath = os.path.join(os.getcwd(), 'history', filename)

    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('')

    if request.method == 'POST':
        with open(filepath, 'a') as f:
            f.write(content + '\n')

    with open(filepath, 'r') as f:
        content = f.read()

    return jsonify({'usrid': usrid, 'content': content}), 200

if __name__ == '__main__':
    app.run()
