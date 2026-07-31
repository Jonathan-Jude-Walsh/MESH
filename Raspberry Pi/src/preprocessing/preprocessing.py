import scipy.signal as signal

def bandpass(
    audio,
    fs,
    low=20,
    high=1600
):

    b,a = signal.butter(

        4,

        [low,high],

        btype="band",

        fs=fs
    )

    return signal.filtfilt(
        b,
        a,
        audio
    )