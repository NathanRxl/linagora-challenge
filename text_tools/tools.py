import nltk
import string
import re


def clean_text(text):
    # convert to lower case
    text = text.lower()

    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in string.punctuation)
    # strip extra white space
    text = re.sub(' +',' ',text)

    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    tokens = text.split(' ')

    if tokens == [""]:
        return " "

    # Stem and remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = nltk.stem.PorterStemmer()
    tokens_stemmed = list()
    for token in tokens:
        if token in stopwords + ["oed", "aed"]:
            continue
        tokens_stemmed.append(stemmer.stem(token))
    tokens = tokens_stemmed

    cleaned_text = ' '.join(tokens)
    # Handle the case when the cleaned text is an empty string
    if cleaned_text == "":
        return " "

    return cleaned_text
