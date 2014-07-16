import pandas as pd
import numpy as np
POS_FILE = 'sentiment_words/positive-words.txt'
NEG_FILE = 'sentiment_words/negative-words.txt'

# chnage this part for different files
INPUT_FILE = 'essay_token_no_lem.csv'
FIELD_NAME = 'tokens'
OUTPUT_FILE = 'sentiment_essays_no_lem.csv'


def read_sentiment_words(file_name):
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = filter(lambda x: len(x) > 1 and not x.startswith(';'), lines)
        lines = [l.strip('\n') for l in lines]
        return frozenset(lines)


def get_sentimant_count(text, pos_words, neg_words):
    tokens = text.split('|')
    total_tokens = len(tokens)
    pos_tokens = 0
    neg_tokens = 0
    for token in tokens:
        if token in pos_words:
            pos_tokens += 1
            #print 'pos: %s' % token
        elif token in neg_words:
            neg_tokens += 1
            #print 'neg: %s' % token

    #print pos_tokens, neg_tokens, total_tokens
    return (pos_tokens, neg_tokens, total_tokens)


def main():
    pos_words = read_sentiment_words(POS_FILE)
    neg_words = read_sentiment_words(NEG_FILE)

    text_df = pd.read_csv(INPUT_FILE)
    text_df = text_df.sort('projectid')

    texts = text_df[FIELD_NAME].fillna('')
    token_cnts = []
    pos_cnts = []
    neg_cnts = []
    for idx, text in enumerate(texts):
        p, n, t = get_sentimant_count(text, pos_words, neg_words)
        pos_cnts.append(p)
        neg_cnts.append(n)
        token_cnts.append(t)
        if idx % 1000 == 0:
            print idx

    output_data = {
        'projectid': text_df['projectid'],
        'pos_cnt': pos_cnts,
        'neg_cnt': neg_cnts,
        'token_cnts': token_cnts,
    }

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print 'done!'

if __name__ == '__main__':
    main()
