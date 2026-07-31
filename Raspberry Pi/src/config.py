MODEL = "rf"

FEATURE_METHOD = "dsp"

WINDOW_SEC = 5

SAMPLE_RATE = 96000

MODEL_PATH = (
    "./trained_models/rf.pkl"
)

CLASS_NAMES = [
    "Cargo",
    "Passengership",
]

ENABLE_LOGGING = True

SAVE_AUDIO = False

SAVE_FEATURES = False