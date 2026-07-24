import os
import tensorflow as tf

def export_quantized_model(keras_model_path, output_tflite_path):
    print(f"Loading full precision model from {keras_model_path}...")
    
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Cannot find trained model at {keras_model_path}. Run training first.")
        
    model = tf.keras.models.load_model(keras_model_path)
    
    print("Configuring TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    print("Converting model. This may take a few minutes...")
    quantized_tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
    
    with open(output_tflite_path, "wb") as f:
        f.write(quantized_tflite_model)
        
    original_size_mb = os.path.getsize(keras_model_path) / (1024 * 1024)
    quantized_size_mb = os.path.getsize(output_tflite_path) / (1024 * 1024)
    
    print(f"Quantization Complete!")
    print(f"Original Model Size:  {original_size_mb:.2f} MB")
    print(f"Quantized Model Size: {quantized_size_mb:.2f} MB")
    print(f"Model saved to: {output_tflite_path}")

if __name__ == "__main__":
    INPUT_KERAS_MODEL = "models/checkpoints/demand_engine_best.keras"
    OUTPUT_TFLITE_MODEL = "models/demand_engine_quant.tflite"
    if not os.path.exists(INPUT_KERAS_MODEL):
        print("Creating dummy checkpoint for testing the exporter...")
        os.makedirs("models/checkpoints/", exist_ok=True)
        inputs = tf.keras.Input(shape=(10,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        dummy_model = tf.keras.Model(inputs, outputs)
        dummy_model.save(INPUT_KERAS_MODEL)
    
    export_quantized_model(INPUT_KERAS_MODEL, OUTPUT_TFLITE_MODEL)