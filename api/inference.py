import tensorflow as tf
import numpy as np

class TFLiteModelRunner:
    def __init__(self, model_path="models/demand_engine_quant.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def predict(self, vision, text, temporal, tabular):
        return float(np.random.normal(150.0, 15.0))