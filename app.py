# importing packages
from flask import Flask, jsonify, request
from gender_identification import gender_predict
import os
import sounddevice as sd
import soundfile as sf

# initialize the flask app
app = Flask(__name__)

# route to record audio
@app.route('/record_audio', methods=['POST'])
def record_audio():
    # Get the duration of the recording
    duration = request.args.get('duration', default=5, type=int)

    # Set the audio parameters
    fs = 44100  # Sample rate
    channels = 1  # Mono

    # Record the audio
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # Wait for the recording to finish

    # Save the audio file
    file_name = 'audio.wav'
    sf.write(file_name, myrecording, fs)

    return jsonify({'message': 'Audio recorded successfully'}), 200

# route to predict gender
@app.route('/predict_gender', methods=['POST'])
def predict_gender():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'File not found'}), 400

    # Save the uploaded file
    file = request.files['file']
    file_path = 'audio.wav'
    file.save(file_path)

    # Predict gender
    gender = gender_predict(file_path)

    # if gender <= 0.5:
    #     response = 'male'
    # else:
    #     response = 'female'

    return jsonify({'gender': gender}), 200


if __name__ == '__main__':
    app.run(debug=True)