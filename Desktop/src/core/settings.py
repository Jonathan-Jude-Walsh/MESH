SETTINGS = {

    "validation": {

        "n_splits": 5,

        "test_size": 0.20,

        "random_seed": 42,

        "split_mode": "groupkfold",
    },

    "training": {

        "epochs": 50,

        "batch_size": 32,

        "learning_rate": 1e-4,

        "patience": 5,

        "validation_split": 0.1,

        "verbose": 1,

        "feature_type": "mfcc",  # logmel, mel, mfcc

        "target_width": 79, 

        "target_height": 32,

        "mobilenet_dropout": 0.30
    },

    "mfcc": {

        "n_mfcc": 13,

        "n_fft": 4096,

        "hop_length": 1024,

        "fmax": 1600,
    },

    "mel": {

        "n_fft": 4096,

        "hop_length": 1024,

        "n_mels": 32,

        "fmax": 1600,
    },

    "demon": {

        "n_fft": 1024,

        "hop_length": 512,

        "nperseg": 512,

        "noverlap": 256,

        "dropout": 0.30,
    },

    "cnn": {

        "filters_1": 16,

        "filters_2": 32,

        "filters_3": 64,

        "dense_units": 64,

        "dropout": 0.30,

        "target_height": 256,

        "target_width": 256
    },

    "mobilenet": {

        "image_size": 96,

        "dense_units": 64,

        "dropout": 0.30,
    },

    "resnet": {

        "dropout": 0.30,
    },

    "hmm": {

        "n_components": 4,

        "n_iter": 100,
    },

    "gmm": {
    
            "n_components": 4,
    
            "n_iter": 100,
        },

    "display": {

        "feature_name_width": 15,
    }
}