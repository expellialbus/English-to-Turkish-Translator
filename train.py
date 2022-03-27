import pathlib

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import pickle

from preprocessing import *
from layers.embedding import PositionalEmbedding
from layers.encoder import TransformerEncoder
from layers.decoder import TransformerDecoder


tf.keras.utils.set_random_seed(42)


def main():
    url = "http://www.manythings.org/anki/tur-eng.zip"
    dataset_path = pathlib.Path("dataset", "tur.txt")

    vocab_size = 30000
    embed_dim = 256
    max_length = 100
    dense_units = 2048
    num_heads = 8
    n = 1
    dropout = 0.5
    num_parallel_calls = 4

    download_dataset(url, "dataset")
    train_pairs, val_pairs, test_pairs = split_dataset(dataset_path)

    source_vectorizer, target_vectorizer = build_vectorizers(
        train_pairs, vocab_size, max_length
    )

    train_ds = create_dataset(
        train_pairs,
        (source_vectorizer, target_vectorizer),
        num_parallel_calls=num_parallel_calls,
    )

    val_ds = create_dataset(
        val_pairs,
        (source_vectorizer, target_vectorizer),
        num_parallel_calls=num_parallel_calls,
    )

    test_ds = create_dataset(
        test_pairs,
        (source_vectorizer, target_vectorizer),
        num_parallel_calls=num_parallel_calls,
    )

    transformer = get_model(
        vocab_size, embed_dim, max_length, dense_units, num_heads, n, dropout
    )

    transformer.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # model results after last epoch of training
    # loss: 0.0607 - accuracy: 0.7840 - val_loss: 0.1576 - val_accuracy: 0.6381
    transformer.fit(train_ds, epochs=10, validation_data=val_ds)

    # model results on test set
    # loss: 0.1584 - accuracy: 0.6382
    transformer.evaluate(test_ds)

    transformer.save("model/translator.h5")


def get_model(
    vocab_size, embed_dim, max_length, dense_units, num_heads, n=1, dropout=0.5
):
    """
    Returns the model which built according as parameters

    Parameters
    ----------
    vocab_size : int
                 Vocabulary size of the dataset

    embed_dim : int
                Output dimension of the embedding layer

    max_length : int
                Maximum length for sentences

    dense_units : int
                Number of units will be used inside dense layers

    num_heads : int
                  Number of heads for MultiHeadAttention

    n : int, default=3
        Number of layers that will be stacked for both encoder and decoder layers

    dropout : float, default=0.5
            Applied dropout level

    Returns
    -------
    An instance of tf.keras.models.Model

    """

    encoder_inputs = Input(shape=(None,), name="english")
    embedding = PositionalEmbedding(vocab_size, embed_dim, max_length)(encoder_inputs)

    encoder_outputs = TransformerEncoder(num_heads, embed_dim, dense_units)(embedding)
    for _ in range(n - 1):
        encoder_outputs = TransformerEncoder(num_heads, embed_dim, dense_units)(
            encoder_outputs
        )

    decoder_inputs = Input(shape=(None,), name="turkish")
    embedding = PositionalEmbedding(vocab_size, embed_dim, max_length)(decoder_inputs)

    decoder_outputs = TransformerDecoder(num_heads, embed_dim, dense_units)(
        embedding, encoder_outputs
    )
    for _ in range(n - 1):
        decoder_outputs = TransformerDecoder(num_heads, embed_dim, dense_units)(
            decoder_outputs, encoder_outputs
        )

    dropout = Dropout(dropout)(decoder_outputs)
    dense_outputs = Dense(vocab_size, activation="softmax")(dropout)

    transformer = Model([encoder_inputs, decoder_inputs], dense_outputs)

    return transformer


if __name__ == "__main__":
    main()
