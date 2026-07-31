import numpy as np
import scipy.signal as signal

def extract(audio,sr):

    envelope = np.abs(
        signal.hilbert(audio)
    )

    f,t,S = signal.spectrogram(

        envelope,

        fs=sr,

        nperseg=512,

        noverlap=256
    )

    return np.log1p(S)