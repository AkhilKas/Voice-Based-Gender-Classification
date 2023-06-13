# importing packages
import pickle
import numpy as np

# importing functions
from audio_features import extract_features

# loading the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# loading the saved model
with open('RandomForestClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

# storing the location of the voice file
# voice_source = input('Enter the location of the voice file: ')

# predicting the gender
def gender_predict(voice_source):

    # extracting and scaling the features
    ft = np.array(extract_features(voice_source))
    ft_scaled = scaler.transform(ft.reshape(1, -1))

    # predicting the gender
    probability = model.predict(ft_scaled)

    if probability <= 0.5:
        return('male')

    else:
        return('female')


# print(gender_predict('C:/Users/Akhilesh/Downloads/st1.wav'))
