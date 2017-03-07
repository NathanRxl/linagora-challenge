import re
import string
from nltk.tag import pos_tag


def truncate_body(body):
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
    body = body[:20]
    if body.find(" ") != -1:
        body = body[::-1].split(" ", 1)[1][::-1]
    return body.lower()


def preprocess_NNP(X_train):
    nnp_set = set()
    for i, body in enumerate(X_train["body_beginning"].tolist()):
        if (i + 1) % 1000 == 0:
            print("{} messages processed, {} unique nnp found".format(i + 1, len(nnp_set)))
        tagged_body = pos_tag(body.split())
        for nnp in [tag[0] for tag in tagged_body if tag[1] == 'NNP' or tag[1] == 'JJ']:
            nnp_set.add(nnp)
    return list(nnp_set)
