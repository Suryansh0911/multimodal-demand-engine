import tensorflow as tf

def build_temporal_encoder(temporal_window=30, embedding_dim=128):
    temporal_input = tf.keras.Input(shape=(temporal_window, 1), name="temporal_input")
    
    rnn_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False)
    )(temporal_input)
    
    temporal_embedding = tf.keras.layers.Dense(embedding_dim, activation='relu')(rnn_out)
    
    return tf.keras.Model(inputs=temporal_input, outputs=temporal_embedding, name="TemporalEncoder")