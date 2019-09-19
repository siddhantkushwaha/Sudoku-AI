import numpy as np

from tensorflow_core.python.keras import models


class Predictor:
    def __init__(self, path='models/digit_rec.h5'):
        self.model = models.load_model(path)

    def predict(self, images):
        return np.asarray([np.argmax(i) for i in self.model.predict(images)])
