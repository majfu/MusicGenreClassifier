from flask import Flask, request, jsonify
from backend.prediction_service import PredictionService
import os
from src.config.config import MODEL_MODE
from backend.path_utils import get_audio_files_dir_path

app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def get_prediction():
    file = request.files['file']

    audio_files_dir = str(get_audio_files_dir_path())
    os.makedirs(audio_files_dir, exist_ok=True)
    file_path = os.path.join(audio_files_dir, file.filename)
    file.save(file_path)

    prediction_service = PredictionService(file_path, MODEL_MODE)
    message, genres_list = prediction_service.get_predictions()

    return build_response(message, genres_list)


def build_response(message, genres_list):
    if message == 'OK':
        return jsonify({"message": message, "genres": genres_list}), 200
    return jsonify({"message": message, "genres": genres_list}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
