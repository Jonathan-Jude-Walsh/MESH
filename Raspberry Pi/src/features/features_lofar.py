import numpy as np
import librosa

def extract(audio,sr):

    S = librosa.stft(

        audio,

        n_fft=1024,

        hop_length=512
    )

    S = np.abs(S)

    return librosa.amplitude_to_db(
        S,
        ref=np.max
    )