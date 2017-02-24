import nltk
import string
import re


# nltk.download('averaged_perceptron_tagger')

def clean_text(text):
    # convert to lower case
    text = text.lower()


    # import pdb
    # pdb.set_trace()
    # text = re.sub('fw:(.*)',' ',text)
    # text = re.sub('fwd:(.*)',' ',text)
    # text = re.sub('-original message-(.*)','',text)


    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in string.punctuation)
    # strip extra white space
    text = re.sub(' +',' ',text)


    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    tokens = text.split(' ')


    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #   string = re.sub(r"\'s", " \'s", string)
    #   string = re.sub(r"\'ve", " \'ve", string)
    #   string = re.sub(r"n\'t", " n\'t", string)
    #   string = re.sub(r"\'re", " \'re", string)
    #   string = re.sub(r"\'d", " \'d", string)
    #   string = re.sub(r"\'ll", " \'ll", string)
    #   string = re.sub(r",", " , ", string)
    #   string = re.sub(r"!", " ! ", string)
    #   string = re.sub(r"\(", " ( ", string)
    #   string = re.sub(r"\)", " ) ", string)
    #   string = re.sub(r"\?", " ? ", string)
    # string = re.sub(r"\s{2,}", " ", string)


    if tokens == [""]:
        return " "

    # apply POS-tagging
    # tagged_tokens = nltk.pos_tag(tokens)
    # tokens_keep = []
    # for i in range(len(tagged_tokens)):
    #     item = tagged_tokens[i]
    #     if (
    #         item[1] == 'NN' or
    #         item[1] == 'NNS' or
    #         item[1] == 'NNP' or
    #         item[1] == 'NNPS' or
    #         item[1] == 'JJ' or
    #         item[1] == 'JJS' or
    #         item[1] == 'JJR'
    #     ):
    #         tokens_keep.append(item[0])
    # tokens = tokens_keep

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
    if cleaned_text == "":
        return " "

    return cleaned_text
