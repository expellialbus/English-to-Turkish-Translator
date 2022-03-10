from zipfile import ZipFile
import pathlib
import requests
import random
import string

from tensorflow.keras.layers import TextVectorization
import tensorflow as tf


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
    train_set : list of tuples
    val_set : list of tuples
    test_set : list of tuples

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

    random.shuffle(text_pairs)

    num_samples = len(text_pairs)
    num_val_samples = int(val_ratio * num_samples)
    num_test_samples = int(test_ratio * num_samples)

    val_pairs = text_pairs[:num_val_samples]
    test_pairs = text_pairs[num_val_samples:num_test_samples]
    train_pairs = text_pairs[num_val_samples + num_test_samples :]

    return train_pairs, val_pairs, test_pairs


def build_vectoizers(train_pairs, vocab_size, max_length):
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

        lowercase = tf.strings.lower(input)
        return tf.strings.regex_replace(lowercase, strip_chars, "")

    source_vectorizer = TextVectorization(
        max_tokens=vocab_size, output_mode="int", output_seqeunce_length=max_length
    )

    target_vectorizer = TextVectorization(
        max_token=vocab_size,
        output_mode="int",
        output_seqeunce_length=max_length + 1,
        standardize=standardization,
    )

    english_texts = [pair[0] for pair in train_pairs]
    turkish_texts = [pair[1] for pair in train_pairs]

    source_vectorizer.adapt(english_texts)
    target_vectorizer.adapt(turkish_texts)

    return source_vectorizer, target_vectorizer
