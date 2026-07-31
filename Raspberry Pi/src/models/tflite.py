import numpy as np
import tensorflow as tf

class TFLiteModel:

    def __init__(
        self,
        path
    ):

        self.interpreter = (
            tf.lite.Interpreter(
                model_path=path
            )
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter
            .get_input_details()
        )

        self.output_details = (
            self.interpreter
            .get_output_details()
        )

    def predict(
        self,
        feature
    ):

        feature = np.expand_dims(
            feature,
            0
        ).astype(
            np.float32
        )

        self.interpreter.set_tensor(

            self.input_details[0]['index'],

            feature
        )

        self.interpreter.invoke()

        output = (
            self.interpreter.get_tensor(

                self.output_details[0]['index']
            )
        )

        return np.argmax(
            output
        )