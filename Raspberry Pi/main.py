from config import *

from audio.recorder import record_audio

from preprocessing.filters import bandpass

from features.dsp import extract

from models.rf import RandomForestModel

model = RandomForestModel(
    MODEL_PATH
)

while True:

    audio = record_audio(

        WINDOW_SEC,

        SAMPLE_RATE
    )

    audio = bandpass(

        audio,

        SAMPLE_RATE
    )

    feature = extract(

        audio,

        SAMPLE_RATE
    )

    pred = model.predict(
        feature
    )

    print(
        "Prediction:",
        pred
    )