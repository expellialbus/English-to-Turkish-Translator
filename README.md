# English to Turkish Translator

This project aims to translate English sentences to their Turkish equivalents via a model which is built upon Transformer Architecture. The model consists of two distinct parts:

- [Transformer Encoder](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/layers/encoder.py)
- [Transformer Decoder](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/layers/decoder.py)

and trained on [this dataset](http://www.manythings.org/anki/tur-eng.zip). The dataset can also be found under the _dataset_ folder of the project.

### Files and folders of the project:

| :warning: **Due to file upload limitations, this project does not contain any saved model file.** |
| ------------------------------------------------------------------------------------------------- |

<br />

> [dataset](https://github.com/recep-yildirim/English-to-Turkish-Translator/tree/master/dataset)

As mentioned above, this folder contains the dataset download from the provided link.

> [layers](https://github.com/recep-yildirim/English-to-Turkish-Translator/tree/master/layers)

This folder contains layers that _tensorflow_ does not contain itself. These layers are:

- [TransformerEncoder](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/layers/encoder.py#L15): Encoder layer of the model which is built with transformer architecture.

- [TransformerDecoder](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/layers/decoder.py#L15): Decoder layer of the model which is built with transformer architecture.

- [PositionalEmbedding](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/layers/embedding.py#L8): Embedding layer that additionally implements positional encodings for input sentences.

> [preprocessing.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py)

Contains codes for downloading and preprocessing the dataset.

> [train.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/train.py)

Contains codes for building and training the the model.

> [inference.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/inference.py)

Contains codes to use the trained model.

<br />

# How Things Work

## What does the [preprocessing.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py) file do?

In short, as the name suggests, the [preprocessing.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py) file contains some preprocess functions for the [dataset](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/dataset/tur-eng.zip).

It offers a [method](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py#L15) to download the [dataset](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/dataset/tur-eng.zip). If you get an error like:

    BadZipFile: File is not a zip file

Just try to change the [**User-Agent**](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py#L39) header from **Mozilla/5.0** to something different. This problem occurs due to some kind of server security feature. To get more information about this problem, see [this link](https://stackoverflow.com/a/16627277).

After complete the dowloading, it decompresses the file and save it to the path specified by a parameter.

[Another function](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py#L51) inside of the [preprocessing.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py) file splits the dataset into a train, validation and test sets and returns them. The ratio of how much of the dataset will be splitted as validation and test sets can be controlled via function parameters.

The [<code>build_vectorizers</code>](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py#L110) function will built vectorizers for both _source sentence_ and _target sentence_ and adapts both vectorizers to its dataset.

At last, the <code>[create_dataset](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/preprocessing.py#L188)</code> function creates the dataset from pairs that sent to the function.

<br />

## What does [train.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/train.py) file do?

[train.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/train.py) file contains functions to build and train according to specified parameters.
The [<code>main</code>](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/train.py#L16) function, first gets the raw texts and adapt a vectorizer on the train part of this texts. Then creates the vectorized datasets. After the dataset creation, it invokes the [<code>get_model</code>](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/train.py#L73) function to build the model (this model can be adjusted according to function parameters. For more detail about parameters, see the _doc string_ of the function). Finally, it trains the model and saves it.

<br />

## What does [inference.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/inference.py) file do?

At last, [inference.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/inference.py) file contains some functions to use the model to make inferences on example inputs. It creates the vectorizers (since the target vectorizer contains user defined standardization function, it could not be saved, this the reason why [inference.py](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/inference.py) has a function to create new vectorizers). After create the vectorizers, loads the model from disk. The [<code>translate</code>](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/inference.py#L79) function is the main function to make inferences and all other test functions (e.g. [<code>test_with_console_input</code>](https://github.com/recep-yildirim/English-to-Turkish-Translator/blob/master/inference.py#L176)) use this function.

    P.S.: Additional information to the above explanations can be found in the source codes.

<br />

# Test Results

## Metrics

\
**Train**

- **Loss:** 0.0607
- **Accuracy:** 0.7840

\
**Validation**

- **Loss:** 0.1576
- **Accuracy:** 0.6381

\
**Test**

- **Loss:** 0.1584
- **Accuracy:** 0.6382

<br />

## Test on the Test Set

<br />

```
English Sentence: Tom and I don't eat out as often as we used to.
Turkish Equivalent: [start] tom ve ben çoğu zaman [UNK] hakkında birlikte yemek yeriz [end]
-----------------------------------------------------------------------------------------------
English Sentence: If you can't read, it's not my fault.
Turkish Equivalent: [start] eğer benim hatam [UNK] değildir [end]
-----------------------------------------------------------------------------------------------
English Sentence: Tom is a former world triathlon champion.
Turkish Equivalent: [start] tom tüm ocak ayı şubat [end]
-----------------------------------------------------------------------------------------------
English Sentence: I hope no one steals my stuff.
Turkish Equivalent: [start] umarım herhangi bir şey [UNK] olmaz [end]
-----------------------------------------------------------------------------------------------
English Sentence: Can we change rooms?
Turkish Equivalent: [start] [UNK] değişim edebilir miyiz [end]
-----------------------------------------------------------------------------------------------
English Sentence: Tom never borrows money from his friends.
Turkish Equivalent: [start] tom arkadaşlarından hiç borç para almaz [end]
-----------------------------------------------------------------------------------------------
English Sentence: I didn't know Tom was allergic to bees.
Turkish Equivalent: [start] tomun [UNK] alerjisi olduğunu bilmiyordum [end]
-----------------------------------------------------------------------------------------------
English Sentence: I can't go swimming today.
Turkish Equivalent: [start] bugün yüzmeye gidemem [end]
-----------------------------------------------------------------------------------------------
English Sentence: Tom was the third victim.
Turkish Equivalent: [start] tom üçüncü [UNK] [end]
-----------------------------------------------------------------------------------------------
English Sentence: I saw Tom walking down the beach.
Turkish Equivalent: [start] tomu plajda kumdan gördüm [end]
-----------------------------------------------------------------------------------------------
```
