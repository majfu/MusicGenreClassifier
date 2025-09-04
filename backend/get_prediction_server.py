from flask import Flask, request

app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def get_prediction():
    request_data = request.get_json()
    files_path = request_data['filePath']

    return 'xxx'


if __name__ == '__main__':
    app.run(debug=True)
