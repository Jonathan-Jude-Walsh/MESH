import librosa
import numpy as np

def extract(audio,sr):

    mel = librosa.feature.melspectrogram(

        y=audio,

        sr=sr
    )

    return librosa.power_to_db(
        mel,
        ref=np.max
    )