# Voice-Based-Gender-Classification

Voice-based gender classification is a process of determining the gender of an individual based on their voice characteristics. It utilizes various techniques from the field of speech processing and machine learning to analyze and extract features from the voice signal that are indicative of the person's gender.

The process involves the following steps:

1. Data Collection: A dataset of voice recordings from individuals of different genders is collected. The dataset should be diverse and representative of the target population.

2. Feature Extraction: Various acoustic features are extracted from the voice recordings, such as fundamental frequency (pitch), spectral characteristics, and temporal patterns. These features capture the unique characteristics of the voice that are correlated with gender differences.

3. Model Training: Machine learning algorithms, such as logistic regression, decision tree classifier, or random forest classifier are trained using the extracted features and the corresponding gender labels from the dataset. The model learns to map the voice features to the corresponding gender categories.

4. Testing and Evaluation: The trained model is then tested on new voice samples to predict the gender of the speaker. The accuracy of the classification is evaluated by comparing the predicted gender with the known ground truth labels.

## [Dataset](https://www.kaggle.com/datasets/primaryobjects/voicegender)
The following acoustic properties of each voice are measured and included within the CSV:

- meanfreq: mean frequency (in kHz)
- sd: standard deviation of frequency
- median: median frequency (in kHz)
- Q25: first quantile (in kHz)
- Q75: third quantile (in kHz)
- IQR: interquantile range (in kHz)
- skew: skewness (skewness is a measure of the asymmetry of a distribution)
- kurt: kurtosis (kurtosis measures the peakedness or flatness of a distribution compared to a normal distribution)
- sp.ent: spectral entropy
- sfm: spectral flatness
- mode: mode frequency
- centroid: frequency centroid (see specprop)
- meanfun: average of fundamental frequency measured across acoustic signal
- minfun: minimum fundamental frequency measured across acoustic signal
- maxfun: maximum fundamental frequency measured across acoustic signal
- meandom: average of dominant frequency measured across acoustic signal
- mindom: minimum of dominant frequency measured across acoustic signal
- maxdom: maximum of dominant frequency measured across acoustic signal
- dfrange: range of dominant frequency measured across acoustic signal
- modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
- label: male or female
