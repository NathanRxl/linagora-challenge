import re
import nltk
import string


def truncate_body(body, char_limit=20):
    if body.find("Original Message") != -1:
        body = body[:body.find("Original Message")]
    if body.find("Forwarded by") != -1:
        body = body[:body.find("Forwarded by")]
    if body.find("X-FileName") != -1:
        body = body[body.find("."):]
        body = body[body.find(" "):]
    body = body.translate(string.punctuation)
    body = re.sub("\.|\,|\-|\;|\(|\)|\:|\!|\<|\>|\+|\"|\?|\$|\_|\*",
                  " ", body)
    body = re.sub("\d+", " ", body)
    body = body[:char_limit]
    if body.find(" ") != -1:
        body = body[::-1].split(" ", 1)[1][::-1]
    return body.lower()


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
