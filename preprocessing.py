from zipfile import ZipFile
import pathlib
import requests
import string
import re

from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import numpy as np


tf.keras.utils.set_random_seed(42)


def download_dataset(url, save_path=""):
    """
    Downloads a given file and saves it to specified path

    Parameters
    ----------
    url : str
          Download source

    save_path : str, pathlib.Path, optional
                File path to save the downloaded file
                If it is not specified, downloaded file will be save current working directory
                Directory does not have to exist

    """

    file_name = url.rsplit("/", maxsplit=1)[1]  # gets the file name from url
    save_path = pathlib.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    save_path /= file_name

    # to learn the answer to why the headers have been defined, see the following link
    # note that the answer has been modified according to requests module
    # https://stackoverflow.com/a/16627277
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    # saves downloaded file
    with open(save_path, "wb") as file:
        file.write(response.content)

    # extracts the downloaded zip file
    with ZipFile(save_path, "r") as compressed:
        compressed.extractall(save_path.parent)


def split_dataset(dataset_path, val_ratio=0.15, test_ratio=0.15):
    """
    Reads the dataset from the specified path and then creates
    both validation set and test set according to specified size

    Parameters
    ----------
    dataset_path: str, pathlib.Path
                  Dataset path which contains the downloaded dataset

    val_ratio : float, default=0.15 (15% of the full dataset)
                Validation set ratio will be used to extract specified proportion
                from the full dataset as a validation set

    test_ratio : float, default=0.15 (15% of the full dataset)
                 Test set ratio will be used to extract specified proportion
                 from the full dataset as a validation set

    Returns
    ------
    train_pairs : list of tuples
    val_pairs : list of tuples
    test_pairs : list of tuples

    Note
    ----
    Returned sets contains both source sentence (english in this case) and
    target sentence (turkish in this case) as a list of tuples

    """

    with open(dataset_path, "r") as dataset:

        # last index contains an empty space
        # because '\n' is the last character of sentences
        text_lines = dataset.read().split("\n")[:-1]

    text_pairs = list()
    for text_line in text_lines:

        # first two elements are english sentence and its corresponding turkish version
        # other remained informations are unnecessary to use
        eng, tur, *_ = text_line.split("\t")
        tur = "[start] " + tur + " [end]"
        text_pairs.append((eng, tur))

    np.random.shuffle(text_pairs)

    num_samples = len(text_pairs)
    num_val_samples = int(val_ratio * num_samples)
    num_test_samples = int(test_ratio * num_samples)

    val_pairs = text_pairs[:num_val_samples]
    test_pairs = text_pairs[num_val_samples : num_val_samples + num_test_samples]
    train_pairs = text_pairs[num_val_samples + num_test_samples :]

    return train_pairs, val_pairs, test_pairs


def build_vectorizers(train_pairs, vocab_size, max_length):
    """
    Builds vectorizers on train data
    It will return two vectorizers, one for source sentences (english in this case)
    and one for target sentences (turkish in this case)

    Parameters
    ----------
    train_pairs : list of tuples
                  List of english and turkish sentences' pairs

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

    strip_chars = string.punctuation.replace("[", "").replace("]", "")

    def standardization(input_str):
        """
        A custom standardization function which will be used within TextVectorization
        It almost does the same thing with the default operation inside TextVectorization
        except it does not remove '[' and ']' sign due to they have been used with "start" and "end" tokens

        Parameters
        ----------
        input_str : str
                    An input from the dataset

        Returns
        -------
        A standardized version of the input string

        Note
        ----
        This function will automatically be called by the TextVectorization layer

        """

        lowercase = tf.strings.lower(input_str)

        # characters should contains escape characters (backslashes)
        # or else the characters will be treated as a regular expression pattern
        # and square brackets come from regular expression rules
        return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

    source_vectorizer = TextVectorization(
        max_tokens=vocab_size, output_mode="int", output_sequence_length=max_length
    )

    target_vectorizer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        # since the target sequence has to be one step ahead, this one step should be added
        output_sequence_length=max_length + 1,
        standardize=standardization,
    )

    english_texts = [pair[0] for pair in train_pairs]
    turkish_texts = [pair[1] for pair in train_pairs]

    source_vectorizer.adapt(english_texts)
    target_vectorizer.adapt(turkish_texts)

    return source_vectorizer, target_vectorizer


def create_dataset(pairs, vectorizers, batch_size=64, num_parallel_calls=None):
    """
    Creates the dataset from passed sentence pairs

    Parameters
    ----------
    pairs : nested iterables
            Collection of source and target sentences

    vectorizers : tuple of TextVectorization object
                  Two TextVectorization object, one for source sentences and one for target sentences respectively

    batch_size : int, default=64
                 Batch size of dataset

    num_parallel_calls : int, optional
                         Number of cores will be used during text vectorization process

    Returns
    -------
    CacheDataset, ({
                    english: (dtype=tf.int64, shape=(None, max_length)),
                    turkish: (dtype=tf.int64, shape=(None, max_length)
                   }, turkish: (dtype=tf.int64, shape=(None, max_length)))

    """

    source_vectorizer, target_vectorizer = vectorizers

    def format_dataset(eng, tur):
        """
        Format dataset for training

        Parameters
        ----------
        eng : numpy.array
              English sentences

        tur : numpy.array
              Turkish sentences


        Returns
        -------
        Dictionary of vectorized source and target sentences, target sentences which one step moved to create labels

        Note
        ----
        This function will automatically be called by the map function of the dataset object

        """

        eng = source_vectorizer(eng)
        tur = target_vectorizer(tur)

        return (
            {
                "english": eng,
                # target sequences' lengths are one token more than source sequences
                # to make both sentences same length, one token should be ignored
                # and this cannot be both first token ("[start]") and another token of sentence except last
                # (the last token may be "[end]" token)
                "turkish": tur[:, :-1]
                # again to make both sentences same length, one token should be ignored
                # but this time (due to this sentence will be used as target)
                # its "[start]" token should be ignored
            },
            tur[:, 1:],
        )

    eng, tur = zip(*pairs)  # returns two tuple

    # if tuple does not converted to list
    # rank >= 1 error will be thrown
    eng = list(eng)
    tur = list(tur)

    dataset = tf.data.Dataset.from_tensor_slices((eng, tur))

    # format dataset function supports batch operations, so batch dataset could be used
    # to make processes much faster
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=num_parallel_calls)

    return dataset.shuffle(2048).prefetch(1).cache()
