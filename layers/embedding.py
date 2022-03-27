import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer


tf.keras.utils.set_random_seed(42)


class PositionalEmbedding(Layer):
    """
    Implementation of both embedding and positional encoding

    Attributes
    ----------
    input_dim : int
                Input dimension of token embedding layer

    output_dim : int
                 Output dimension of both token embedding and position embedding layers

    sequence_length : int
                      Input dimension of position embedding layer

    Methods
    -------
    call(self, inputs)
        Function used for computation, see keras docs for more information

    get_config(self)
        See https://keras.io/api/models/model_saving_apis/#getconfig-method

    Notes
    -----
    Also see https://keras.io/guides/making_new_layers_and_models_via_subclassing/
    and https://www.tensorflow.org/guide/keras/masking_and_padding#supporting_masking_in_your_custom_layers

    """

    def __init__(self, input_dim, output_dim, sequence_length, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length

        self.token_embeddings = Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):
        positions = tf.range(self.sequence_length)

        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)

        return embedded_tokens + embedded_positions

    # see https://www.tensorflow.org/guide/keras/masking_and_padding#supporting_masking_in_your_custom_layers
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    # To be able save model, get_config method has to be implemented
    # see https://keras.io/api/models/model_saving_apis/#getconfig-method
    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "sequence_length": self.sequence_length,
            }
        )

        return config
