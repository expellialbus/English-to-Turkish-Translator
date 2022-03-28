import pathlib

import tensorflow as tf
import numpy as np

from preprocessing import *
from layers.embedding import PositionalEmbedding
from layers.encoder import TransformerEncoder
from layers.decoder import TransformerDecoder

tf.keras.utils.set_random_seed(42)


def get_vectorizers(dataset_path, train_pairs, vocab_size, max_length):
    """
    Creates the vectorizers

    Parameters
    ----------
    dataset_path: str, pathlib.Path
                  Dataset path which contains the downloaded dataset

    train_pairs : list of tuples
                  The sentence pairs has been used to create train set

    vocab_size : int
                 Vocabulary size used by TextVectorization

    max_length : int
                 Maximum length of a sentence (will be used for both source and target sentence)

    Returns
    -------
    source_vectorizer : tf.keras.layers.TextVectorization
                        An instance of TextVectorization layer which is adapted to the source sentences (english in this case)

    target_vectorizer : tf.keras.layers.TextVectorization
                        An instance of TextVectorization layer which is adapted to the target sentences (turkish in this case)

    """

    source_vectorizer, target_vectorizer = build_vectorizers(
        train_pairs, vocab_size, max_length
    )

    return source_vectorizer, target_vectorizer


def load_model(model_path, custom_objects=None):
    """
    Loads the model from disk

    Parameters
    ----------
    model_path : str, pathlib.Path
                 The path where the model has been saved

    custom_objects : dict, default=None
                     Custom objects mapping for the models
                     which has been saved as HDF5 format

    Returns
    -------
    model : keras.engine.functional.Functional
            An instance of Keras' Functional Model

    Notes
    -----
    For more information about custom_objects parameter
    see https://www.tensorflow.org/tutorials/keras/save_and_load#saving_custom_objects

    """

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    return model


def translate(model, source_sentence, source_vectorizer, target_vectorizer, max_length):
    """
    Predicts the target sentence as vector and then decodes it to the human readable format

    Parameters
    ----------
    model : tf.keras.Model
            Translator model

    source_sentence : str
                      The source sentence (i.e. the sentence will be translated, english in this case)

    source_vectorizer : tf.keras.layers.TextVectorization
                        An instance of TextVectorization layer which is adapted to the source sentences (english in this case)

    target_vectorizer : tf.keras.layers.TextVectorization
                        An instance of TextVectorization layer which is adapted to the target sentences (turkish in this case)

    max_length : int
                 Maximum length of the translated sentence


    Returns
    -------
    target_sentence : str
                      The translated sentence (turkish in this case)

    """

    vocabulary = target_vectorizer.get_vocabulary()
    index_lookup = dict(enumerate(vocabulary))  # creates the vocabulary lookup

    source_vector = source_vectorizer([source_sentence])
    target_sentence = "[start]"

    for i in range(max_length):
        # since the target vectorizer creates vector that one step ahead (i.e. one additional token)
        # the last token should be eliminated
        target_vector = target_vectorizer([target_sentence])[:, :-1]

        # model has two input layer
        predictions = model.predict([source_vector, target_vector])

        # the prediction shape is [1, max_length, vocab_size]
        # since the predicted token is added to the target sentence
        # at every iteration the last token (i.e. the loop index) in the predictions should be taken
        index = np.argmax(predictions[0, i, :])
        word = index_lookup[index]

        target_sentence += " " + word

        # terminates the loop
        # if the target sentence is completed
        if word == "[end]":
            break

    return target_sentence


def test_with_test_set(
    model, test_set, source_vectorizer, target_vectorizer, max_length=100
):
    """
    Tests the model with a test set

    Parameters
    ----------
    model : tf.keras.Model
            Translator model

    test_set : list of tuples
               The test set

    source_vectorizer : tf.keras.layers.TextVectorization
                        An instance of TextVectorization layer which is adapted to the source sentences (english in this case)

    target_vectorizer : tf.keras.layers.TextVectorization
                        An instance of TextVectorization layer which is adapted to the target sentences (turkish in this case)

    max_length : int, default=100
                 Maximum length of the translated sentence

    """
    np.random.shuffle(test_set)

    source_sentences = [pair[0] for pair in test_set]

    for source_sentence in source_sentences[:10]:
        target_sentence = translate(
            model, source_sentence, source_vectorizer, target_vectorizer, max_length
        )

        print(f"English Sentence: {source_sentence}")
        print(f"Turkish Equivalent: {target_sentence}")
        print("-" * 100)


def test_with_console_input(
    model, source_vectorizer, target_vectorizer, max_length=100
):
    """
    Tests the model with the console inputs

    Parameters
    ----------
    model : tf.keras.Model
            Translator model

    source_vectorizer : tf.keras.layers.TextVectorization
                        An instance of TextVectorization layer which is adapted to the source sentences (english in this case)

    target_vectorizer : tf.keras.layers.TextVectorization
                        An instance of TextVectorization layer which is adapted to the target sentences (turkish in this case)

    max_length : int, default=100
                 Maximum length of the translated sentence

    """
    source_sentence = input("Enter the english sentence: ")
    target_sentence = translate(
        model, source_sentence, source_vectorizer, target_vectorizer, max_length
    )

    print(f"Turkish Equivalent: {target_sentence}")


def main():
    dataset_path = pathlib.Path("dataset", "tur.txt")
    model_path = pathlib.Path("model", "translator.h5")

    vocab_size = 30000
    max_length = 100

    train_pairs, _, test_pairs = split_dataset(dataset_path)

    source_vectorizer, target_vectorizer = get_vectorizers(
        dataset_path, train_pairs, vocab_size, max_length
    )

    custom_objects = {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder,
    }

    model = load_model(model_path, custom_objects)
    test_with_test_set(model, test_pairs, source_vectorizer, target_vectorizer)


if __name__ == "__main__":
    main()
