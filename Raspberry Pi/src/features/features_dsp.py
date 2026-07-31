import numpy as np
import librosa

def extract(audio,sr):

    rms = np.mean(
        librosa.feature.rms(
            y=audio
        )
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(
            audio
        )
    )

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )
    )

    return np.array([
        rms,
        zcr,
        centroid,
        rolloff
    ])