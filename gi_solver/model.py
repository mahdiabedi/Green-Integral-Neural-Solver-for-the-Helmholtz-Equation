import tensorflow as tf
import keras

class EmbedderLayer(tf.keras.layers.Layer):#The sinusoidal embedder layer
    def __init__(self, domain_bounds, **kwargs):
        super(EmbedderLayer, self).__init__(**kwargs)
        self.domain_bounds = domain_bounds  # Not used here

    @tf.function()
    def call(self, inputs):
        
        #option 1, using 2^K x 
        input1 =  (inputs)
        input2 = tf.math.multiply(input1 , 2.0)
        input4 = tf.math.multiply(input1 , 4.0)
        input8 = tf.math.multiply(input1 , 8.0)
                
        input_all = tf.concat([input1,input2,input4,input8], axis=1)

        # Apply sine and cosine functions
        sin_embed = tf.sin(input_all)
        cos_embed = tf.cos(input_all)

        # Concatenate original input, sine, and cosine embeddings
        output = tf.concat([inputs,sin_embed, cos_embed], axis=1)
        return output
    def get_config(self):
        config = super(EmbedderLayer, self).get_config()
        config.update({"domain_bounds": self.domain_bounds})
        return config

def make_u_model(
    neurons, 
    activation=tf.math.sin, 
    activation_penultima=tf.math.sin, 
    neurons_final=None, 
    dtype=tf.float32, 
    trainableLastLayer=True,
    seed=1234,
    domain_bounds=None):
    
    if neurons_final is None:
        neurons_final = neurons

    b_init = keras.initializers.Zeros()
    
    # --- Helper function for dynamic initialization ---
    def get_init(layer_seed):
            return keras.initializers.GlorotNormal(seed=layer_seed)

    # Input layer
    l0 = keras.layers.Input(shape=(2,), name="x_input", dtype=dtype)
    
    # Apply the embedding layer 
    l10 = EmbedderLayer(name="embedder", domain_bounds=domain_bounds)(l0)
    
    # First dense layer
    l11 = keras.layers.Dense(neurons, activation=activation, dtype=dtype,
                              kernel_initializer=keras.initializers.GlorotUniform(seed=seed), 
                              bias_initializer=b_init, name="layer_1")(l10)
    
    # Second dense layer
    l12 = keras.layers.Dense(neurons, activation=activation, dtype=dtype, name="layer_2",
                              kernel_initializer=get_init(seed + 1),
                              bias_initializer=b_init)(l11)
    
    # Third dense layer
    l13 = keras.layers.Dense(neurons, activation=activation, dtype=dtype, name="layer_3",
                              kernel_initializer=get_init(seed + 2),
                              bias_initializer=b_init)(l12)
    
    # Fourth dense layer
    l14 = keras.layers.Dense(neurons, activation=activation, dtype=dtype, name="layer_4",
                              kernel_initializer=get_init(seed + 5),
                              bias_initializer=b_init)(l13)
                             
    # Penultimate layer
    l1 = keras.layers.Dense(neurons_final, activation=activation_penultima, dtype=dtype, name="penultimate_layer",
                            kernel_initializer=get_init(seed + 3),
                            bias_initializer=b_init)(l14)
    
    # Output layer
    output = keras.layers.Dense(2, use_bias=False, trainable=trainableLastLayer, dtype=dtype, name='Output_layer',
                                kernel_initializer=get_init(seed + 4))(l1)  
    
    # Define model
    u_model = keras.Model(inputs=l0, outputs=output)
    
    return u_model
