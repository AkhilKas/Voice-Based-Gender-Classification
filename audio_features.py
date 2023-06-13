import librosa
import os
import re
import numpy as np
import scipy.stats as st
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def extract_features(audio_path):
    """
    Extracts the specified audio features from the given audio file.

    Args:
        audio_file (str): The path to the audio file to extract features from.

    Returns:    
        features: A lsit containing the extracted features.

    Features include: "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", "sfm", "mode", "centroid", "meanfun","minfun", "maxfun", "meandom", "mindom", "maxdom", "dfrange", "modindx", "label"
    """
    # taking input as .wav file
    # audio_path = input('Enter the path for the .wav file: ')

    # Load audio file
    y, sr = librosa.load(audio_path)

    # Calculate the spectral centroid
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

    # pitch
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nan_to_num(list(f0), nan=0.0)

    # Extract a feature vector
    mfccs = librosa.feature.mfcc(y=y, sr=sr)

    # Extract features
    features = []

    meanfreq = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    features.append(meanfreq)

    sd = np.std(y)
    features.append(sd)

    median = np.median(y)
    features.append(median)

    q25 = np.median(centroids[:, 1])
    features.append(q25)

    q75 = np.median(centroids[:, 3])
    features.append(q75)

    iqr = np.median(centroids[:, 3] - centroids[:, 1])
    features.append(iqr)

    skew = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)[0]
    # skew = st.skew(mfccs)
    features.append(skew)

    kurt = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)[1]
    # kurt = st.kurtosis(mfccs)
    features.append(kurt)

    # Compute power spectrum
    S = np.abs(librosa.stft(y))**2

    eps = 1e-10 # small epsilon values

    # Normalize power spectrum
    S /= (np.sum(S, axis=0) + eps)

    S += eps

    # Compute spectral entropy
    sp_ent = -np.sum(S * np.log2(S), axis=0)
    features.append(sp_ent.mean())

    sfm = librosa.feature.spectral_flatness(y=y)
    features.append(np.mean(sfm))

    mode = st.mode(mfccs, keepdims=True)
    features.append(mode[0].mean())

    features.append(np.mean(centroids))

    meanfun = librosa.feature.mfcc(y=y, sr=sr).mean(axis=1)[1]
    features.append(meanfun)

    minfun = np.min(pitch)
    features.append(minfun)

    maxfun = np.max(pitch)
    features.append(maxfun)

    meandom = librosa.feature.mfcc(y=y, sr=sr).mean(axis=1)[3]
    features.append(meandom)

    # Compute minimum pitch
    mindom = min(f0[voiced_flag])
    features.append(mindom)

    # Compute maximum pitch
    maxdom = max(f0[voiced_flag])
    features.append(maxdom)

    # Compute pitch range
    dfrange = maxdom - mindom
    features.append(dfrange)

    modindx = librosa.feature.mfcc(y=y, sr=sr).mean(axis=1)[12]
    features.append(modindx)

    return features

# scaling the list 
# ft = np.array(extract_features())
# res = scaler.fit_transform(ft[:, np.newaxis])
# print(res)


# print(extract_features())