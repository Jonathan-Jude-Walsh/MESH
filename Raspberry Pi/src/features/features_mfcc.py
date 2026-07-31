import librosa

def extract(audio,sr):

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=sr,

        n_mfcc=13
    )

    return mfcc