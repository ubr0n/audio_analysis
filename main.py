#### Dependencies ####

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from glob import glob
import joblib
import sounddevice as sd
from scipy.io.wavfile import write

filepath1 = "/Users/norbulama/Desktop/CS-4000/audios/lie/"
filepath2 = "/Users/norbulama/Desktop/CS-4000/audios/truth"

audiopath1 = glob(filepath1 + "/*.wav")
audiopath2 = glob(filepath2 + "/*.wav")

fs = 22050  # Sample rate
seconds = 5  # Duration of recording
print("Record the first sample")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('/Users/norbulama/Desktop/CS-4000/TruthOrLieAutomation/userInput/output1.wav', fs,
      myrecording)  # Save as WAV file

print("Record the second sample")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('/Users/norbulama/Desktop/CS-4000/TruthOrLieAutomation/userInput/output2.wav', fs,
      myrecording)  # Save as WAV file

print("Record the third sample")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('/Users/norbulama/Desktop/CS-4000/TruthOrLieAutomation/userInput/output3.wav', fs,
      myrecording)  # Save as WAV file

userInput_path = '/Users/norbulama/Desktop/CS-4000/TruthOrLieAutomation/userInput/'
useraudiopath1 = glob(userInput_path + "/*.wav")

user_mfcc = []
user_mel = []
user_stft = []


for i in useraudiopath1:
    audio, sample_rate = librosa.load(i)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    mel_spectogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=512,
                                                    n_mels=10)  # mel spectogram
    mel_spectogram_processed = np.mean(mel_spectogram.T, axis=0)

    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_stft_processed = np.mean(chroma_stft.T, axis=0)

    user_mfcc.append(mfccs_processed)
    user_mel.append(mel_spectogram_processed)
    user_stft.append(chroma_stft_processed)


modelmfcc = joblib.load('mfccmodel.sav')
modelmel = joblib.load('melspecmodel.sav')
modelstft = joblib.load('chromastftmodel.sav')

mfcc_predict = modelmfcc.predict(user_mfcc)
mel_predict = modelmel.predict(user_mel)
stft_predict = modelstft.predict(user_stft)

print(mfcc_predict)
print(mel_predict)
print(stft_predict)

