import tensorflow as tf

def build_vision_encoder(input_shape=(224, 224, 3), embedding_dim=128):
    vision_input = tf.keras.Input(shape=input_shape, name="vision_input")
    
    base_cnn = tf.keras.applications.EfficientNetV2B0(include_top=False, pooling='avg')
    base_cnn.trainable = False
    
    x = base_cnn(vision_input)
    vision_embedding = tf.keras.layers.Dense(embedding_dim, activation='relu')(x)
    
    return tf.keras.Model(inputs=vision_input, outputs=vision_embedding, name="VisionEncoder")