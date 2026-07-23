import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

# Configuration Paths
ABO_IMG_DIR = "data/raw/abo/images/"
OUTPUT_DIR = "data/processed/"
TOKENIZER_PATH = "C:/Users/surya/OneDrive/Desktop/multimodal-demand-engine/models/local_hf_weights/distilbert-base-uncased"
TFRECORD_SHARDS = 10

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def load_and_preprocess_image(image_path):
    """Loads a JPEG, resizes to 224x224, and returns raw bytes."""
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.uint8)
    return tf.io.encode_jpeg(image).numpy()

def generate_tfrecords():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Local Tokenizer
    print("Loading local HuggingFace tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # 2. Mock Data Merging (Replace with actual Pandas merges of M5 and ABO data)
    # In reality, you will merge M5's sales_train_evaluation.csv with ABO's listings.json
    print("Fusing M5 Time-Series with ABO Metadata...")
    total_samples = 1000 # Dummy count for execution
    
    samples_per_shard = total_samples // TFRECORD_SHARDS
    
    for shard_id in range(TFRECORD_SHARDS):
        shard_path = os.path.join(OUTPUT_DIR, f"multimodal_demand_{shard_id:02d}.tfrecord")
        
        with tf.io.TFRecordWriter(shard_path) as writer:
            for i in range(samples_per_shard):
                
                # --- A. Vision Modality ---
                # Fallback to a blank image if product image is missing
                dummy_image = tf.io.encode_jpeg(tf.zeros([224, 224, 3], dtype=tf.uint8)).numpy()
                
                # --- B. Text Modality ---
                product_title = "Example product title description for text encoder"
                tokens = tokenizer(
                    product_title, 
                    max_length=128, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors="np" # Changed 'tf' to 'np' (NumPy)
                )
                # Explicitly convert the NumPy array to a TensorFlow integer tensor
                input_tensor = tf.convert_to_tensor(tokens['input_ids'][0], dtype=tf.int32)
                text_input_ids = tf.io.serialize_tensor(input_tensor).numpy()
                
                # --- C. Temporal Modality ---
                historical_sales = np.random.uniform(0, 50, size=(30,)).astype(np.float32).tolist()
                
                # --- D. Tabular Modality ---
                # Fusing pricing vectors with unsupervised customer segmentation clusters 
                # (e.g., K-Means/DBSCAN cluster IDs) provides strong behavioral signals.
                tabular_features = np.random.uniform(0, 1, size=(15,)).astype(np.float32).tolist()
                
                # --- E. Target Label ---
                future_demand = [np.random.uniform(10, 100)] # Day 31 demand

                # Construct TFRecord Feature Dictionary
                feature = {
                    'image': _bytes_feature(dummy_image),
                    'text': _bytes_feature(text_input_ids),
                    'historical_sales': _float_feature(historical_sales),
                    'tabular_features': _float_feature(tabular_features),
                    'future_demand': _float_feature(future_demand)
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                
        print(f"Wrote {samples_per_shard} records to {shard_path}")
        
    print("TFRecord generation complete. Ready for training pipeline.")

if __name__ == "__main__":
    generate_tfrecords()