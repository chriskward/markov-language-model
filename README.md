## A Character Based Language Model

### pip install git+https://github.com/chriskward/markov-language-model

<br>

***

This package contains a simple character based 'language model'. The 
**MarkovModel** class learns the parameters of an n-order
markov chain over sequences of characters using a training set
of text.

This can be used for classification or text generation. A trained
model can return log-probability that a test sample
was generated from the same distribution as the training text and
can generate text samples by traversing the markov chain.

<br>

[**An example of text classification** using the UCI ML Repository
Victorian Authors Dataset](victorian-authors-classification.ipynb)

[**The 'Boris-Bot'** - Using a MarkovModel trained on a large corpus
of webscraped Telegraph newpaper articles written by Boris Johnson](text_generation.ipynb)

<br>

***

<br>

This formed part of a masters degree thesis studing the applicability
of a range of machine learning techniques to Plagiarism Detection.
This research question is discussed in a little more detail [here](https://github.com/chriskward/stylometry-toolkit).

Traditional approaches to Natural Language Processing and Stylometry (the
statistical analysis of writing style) typically require extensive
feature engineering. Much attention must be paid to text vectorisation to
ensure a set of relevant features are extracted from text. This approach
is detailed in [stylometry-toolkit](https://github.com/chriskward/stylometry-toolkit).

More recent approaches to NLP directly work with text as a linear sequence of
characters, words or sub-word tokens rather than using the [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model)
methodology. This 'sequence modelling' approach to text is at the heart of
the more recent advances in NLP such effective machine translation and 
realistic text generation using Transformer encoder/decoders.

The **MarkovModel** class, takes a (large) string of training text and uses numpy's
lib.stride_tricks and direct buffer access to view to string as a sequence of
*context_vectors* of length n and *trailing_characters*. For example:

    'Hello' -> ['Hell' , 'o']

By counting the occurances of each *context_vector* and
*trailing_character* combination in the training text the model can estimate probabilites:


$\mathbb{P}( x_i | x_{i-1} , x_{i-2} , ... , x_{i-n})$

    
For example:

$\mathbb{P}(\ \text{'O'}\ |\ \text{'Hell'}\ )$





