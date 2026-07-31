import joblib

class RandomForestModel:

    def __init__(
        self,
        path
    ):

        self.model = joblib.load(
            path
        )

    def predict(
        self,
        feature
    ):

        return self.model.predict(
            feature.reshape(1,-1)
        )[0]