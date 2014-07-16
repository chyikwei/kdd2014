import re
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

INPUT_FILE_NAME = 'raw_data/essays.csv'

OUTPUT_FILE_NAME = 'essay_tokens_test.csv'
DATA_FIELD = 'essay'
NEW_FIELD = 'essay_token'


def text_process(text):
    if pd.isnull(text):
        return ''

    tokens = word_tokenize(text)
    # lower_case
    tokens = [t.lower() for t in tokens]
    # filter out special word
    tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]
    # remove stop words
    tokens = [t for t in tokens if t not in en_stopwords]
    # lemmatizer
    tokens = [lemmatizer.lemmatize(t, wordnet.VERB) for t in tokens]
    tokens = [lemmatizer.lemmatize(t, wordnet.NOUN) for t in tokens]

    # convert to string with '|' separate
    token_text = '|'.join(tokens)

    return token_text


def main():    
    essays = pd.read_csv(INPUT_FILE_NAME)
    essays = essays.sort('projectid')
    essays[NEW_FIELD] = essays[DATA_FIELD].apply(text_process)
    df = pd.DataFrame({'projectid': essays['projectid'], NEW_FIELD: essays[NEW_FIELD]})
    df.to_csv(OUTPUT_FILE_NAME, index=False)


if __name__ == '__main__':
    main()