import xgboost as xgb
import numpy as np

class XGBModel:

    def __init__(
        self,
        path
    ):

        self.model = xgb.XGBClassifier()

        self.model.load_model(
            path
        )

    def predict(
        self,
        feature
    ):

        return self.model.predict(

            feature.reshape(1,-1)

        )[0]