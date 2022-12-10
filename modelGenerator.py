#### Dependencies ####

import joblib
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

from sklearn.metrics import classification_report

filepath1 = "/Users/norbulama/Desktop/CS-4000/audios/lie/"
filepath2 = "/Users/norbulama/Desktop/CS-4000/audios/truth"

audiopath1 = glob(filepath1 + "/*.wav")
audiopath2 = glob(filepath2 + "/*.wav")


def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    mel_spectogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=512,
                                                    n_mels=10)  # mel spectogram
    mel_spectogram_processed = np.mean(mel_spectogram.T, axis=0)

    tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)  # tonal centroid feature
    tonnetz_processed = np.mean(tonnetz.T, axis=0)

    rms = librosa.feature.rms(y=audio)  # root mean square
    rms_processed = np.mean(rms.T, axis=0)

    cent = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)  # spectral centroid
    cent_processed = np.mean(cent.T, axis=0)

    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_stft_processed = np.mean(chroma_stft.T, axis=0)
    librosa.display.specshow(chroma_stft, x_axis='chroma', y_axis='time', sr=sample_rate)
    # plt.show()

    return [mfccs_processed, mel_spectogram_processed, tonnetz_processed, rms_processed, cent_processed,
            chroma_stft_processed]


mfcc_features = []
melspectogram_features = []
tonnetz_features = []
rms_features = []
cent_features = []
chroma_stft_features = []

# Iterate through each sound file and extract the features
for i in audiopath1:
    data = extract_features(i)
    mfcc_features.append([data[0], "lie"])
    melspectogram_features.append([data[1], "lie"])
    tonnetz_features.append([data[2], "lie"])
    rms_features.append([data[3], "lie"])
    cent_features.append([data[4], "lie"])
    chroma_stft_features.append([data[5], "lie"])

    # melspectogram_features.append([mel, 'lie'])
for i in audiopath2:
    data = extract_features(i)
    mfcc_features.append([data[0], "truth"])
    melspectogram_features.append([data[1], "truth"])
    tonnetz_features.append([data[2], "truth"])
    rms_features.append([data[3], "truth"])
    cent_features.append([data[4], "truth"])
    chroma_stft_features.append([data[5], "truth"])

# Convert into a Panda dataframe
mfccfeaturesdf = pd.DataFrame(mfcc_features, columns=['feature', 'class_label'])
melspectogram_featuresdf = pd.DataFrame(melspectogram_features, columns=['feature', 'class_label'])
tonnetz_featuresdf = pd.DataFrame(tonnetz_features, columns=['feature', 'class_label'])
rms_featuresdf = pd.DataFrame(rms_features, columns=['feature', 'class_label'])
cent_featuresdf = pd.DataFrame(cent_features, columns=['feature', 'class_label'])
chroma_stft_featuresdf = pd.DataFrame(chroma_stft_features, columns=['feature', 'class_label'])


# mfccfeaturesdf.head()


def mfccmodel():
    # changing features and class label int numpy array
    X = np.array(mfccfeaturesdf['feature'].tolist())
    y = np.array(mfccfeaturesdf['class_label'].tolist())

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
    classifier.fit(X_train, y_train)

    filename = 'mfccmodel.sav'
    joblib.dump(classifier, filename)

    print(classifier.fit(X_train, y_train))

    y_predict = classifier.predict(X_test)
    print(classification_report(y_test, y_predict))

    c_matrix = metrics.confusion_matrix(y_test, y_predict)

    print("Confusion matrix for mfcc model\n", c_matrix)


def melspecmodel():
    X = np.array(melspectogram_featuresdf['feature'].tolist())
    y = np.array(melspectogram_featuresdf['class_label'].tolist())

    # Train test saplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
    classifier.fit(X_train, y_train)

    filename = 'melspecmodel.sav'
    joblib.dump(classifier, filename)

    print(classifier.fit(X_train, y_train))

    y_predict = classifier.predict(X_test)
    print("Test result based on mel Spectogram\n")
    print(classification_report(y_test, y_predict))

    c_matrix = metrics.confusion_matrix(y_test, y_predict)

    print("Confusion matrix for mel cpectrogram model\n", c_matrix)


def tonnetzmodel():
    X = np.array(tonnetz_featuresdf['feature'].tolist())
    y = np.array(tonnetz_featuresdf['class_label'].tolist())

    # Train test saplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    classifier = svm.SVC(kernel='rbf', gamma='auto', C=2)
    classifier.fit(X_train, y_train)

    filename = 'tonnetzmodel.sav'
    joblib.dump(classifier, filename)

    print(classifier.fit(X_train, y_train))

    y_predict = classifier.predict(X_test)
    print("Test result based on mel tonnetz\n")
    print(classification_report(y_test, y_predict))


def rmsmodel():
    X = np.array(rms_featuresdf['feature'].tolist())
    y = np.array(rms_featuresdf['class_label'].tolist())

    # Train test saplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
    classifier.fit(X_train, y_train)

    filename = 'rmsmodel.sav'
    joblib.dump(classifier, filename)

    print(classifier.fit(X_train, y_train))

    y_predict = classifier.predict(X_test)
    print("Test result based on rms\n")
    print(classification_report(y_test, y_predict))


def centmodel():
    X = np.array(cent_featuresdf['feature'].tolist())
    y = np.array(cent_featuresdf['class_label'].tolist())

    # Train test saplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
    classifier.fit(X_train, y_train)

    filename = 'centmodel.sav'
    joblib.dump(classifier, filename)

    print(classifier.fit(X_train, y_train))

    y_predict = classifier.predict(X_test)
    print("Test result based on cent\n")
    print(classification_report(y_test, y_predict))


def chromastftmodel():
    X = np.array(chroma_stft_featuresdf['feature'].tolist())
    y = np.array(chroma_stft_featuresdf['class_label'].tolist())

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
    classifier.fit(X_train, y_train)

    filename = 'chromastftmodel.sav'
    joblib.dump(classifier, filename)

    print(classifier.fit(X_train, y_train))

    y_predict = classifier.predict(X_test)
    print("Test result based on chromastftmodel\n")
    print(classification_report(y_test, y_predict))

#confusion matrix
    c_matrix = metrics.confusion_matrix(y_test, y_predict)

    print("Confusion matrix for chromastftmodel \n", c_matrix)


mfccmodel()
melspecmodel()
# rmsmodel()
# tonnetzmodel()
# centmodel()
chromastftmodel()
