import sounddevice as sd
import numpy as np

def record_audio(
    duration,
    sample_rate
):

    audio = sd.rec(

        int(
            duration *
            sample_rate
        ),

        samplerate=sample_rate,

        channels=1,

        dtype=np.float32
    )

    sd.wait()

    return audio.flatten()