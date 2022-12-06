import pandas as pd
import numpy as np
import emoji
import wordsegment
from utils import pad_sents, get_mask, get_lens, text_processor
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

wordsegment.load()


def process_tweets(tweets):
    # Process tweets
    tweets = emoji2word(tweets)
    tweets = url2URL(tweets)
    tweets = replace_rare_words(tweets)
    tweets = use_ekphrasis(tweets)
    tweets = remove_useless_punctuation(tweets)
    tweets = np.array(tweets)
    return tweets


def use_ekphrasis(sents):
    for i, sent in enumerate(sents):
        sent = " ".join(text_processor.pre_process_doc(sent))
        sent = sent.replace('<elongated> ', '')
        sent = sent.replace('</elongated> ', '')
        sents[i] = sent
    return sents


def remove_stopwords(sents):
    stop_words = stopwords.words('english')
    for i, sent in enumerate(sents):
        sent = sent.split()
        sent = [w for w in sent if w not in stop_words]
        sents[i] = ' '.join(sent)
    return sents


def emoji2word(sents):
    return [emoji.demojize(str(sent)) for sent in sents]


def remove_useless_punctuation(sents):
    for i, sent in enumerate(sents):
        sent = sent.replace(':', ' ')
        sent = sent.replace('_', ' ')
        sent = sent.replace('...', '.')
        sent = sent.replace('非原创：', '')
        sent = sent.replace('RT ', '')
        sent = re.sub(r'&[\s\S]+;', ' ', sent)
        sent = re.sub('[' + r"""#$%&()*+-/<=>[\]^_`{|}~↘️""" + ']', ' ', sent)
        sent = re.sub('\\b[0-9]+\\b', '', sent)
        sents[i] = sent
    return sents


def remove_replicates(sents):
    for i, sent in enumerate(sents):
        if sent.find('@USER') != sent.rfind('@USER'):
            sents[i] = sent.replace('@USER', '')
            sents[i] = '@USERS ' + sents[i]
    return sents


def replace_rare_words(sents):
    rare_words = {
        'URL': 'http',
        '&amp;': 'and',
    }
    for i, sent in enumerate(sents):
        for w in rare_words.keys():
            sents[i] = sent.replace(w, rare_words[w])
    return sents


def segment_hashtag(sents):
    # E.g. '#LunaticLeft' => 'lunatic left'
    for i, sent in enumerate(sents):
        sent_tokens = sent.split(' ')
        for j, t in enumerate(sent_tokens):
            if t.find('#') == 0:
                sent_tokens[j] = ' '.join(wordsegment.segment(t))
        sents[i] = ' '.join(sent_tokens)
    return sents


def url2URL(sents):
    for i, sent in enumerate(sents):
        urls = re.findall(r'(https?://\S+)', sent)
        for url in urls:
            sents[i] = sent.replace(url, 'URL')
    return sents


def user2USER(sents):
    for i, sent in enumerate(sents):
        users = re.findall(r'(@\w+)', sent)
        for user in users:
            sents[i] = sent.replace(user, '@USER')
    return sents


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


def mydata(path, tokenizer, data_type='train', truncate=512, test_size=0.3):
    data = pd.read_csv(path)
    purl = data['user_id'].values
    tweets = np.array(data['content'].values)
    labels = np.array(data['is_off'].values)
    nums = len(data)

    # Process tweets
    tweets = process_tweets(tweets)
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate, truncation=True)
                 for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    purl_train, purl_test, token_ids_train, token_ids_test, lens_train, lens_test, mask_train, mask_test, labels_train,\
    labels_test = train_test_split(purl, token_ids, lens, mask, labels, test_size=test_size, random_state=0)

    if data_type == 'train':
        return purl_train, token_ids_train, lens_train, mask_train, labels_train
    elif data_type == 'test':
        return purl_test, token_ids_test, lens_test, mask_test, labels_test
    elif data_type == 'ngrams':
        model = TfidfVectorizer(analyzer='char',
                                ngram_range=(2, 2),
                                max_features=10000,
                                use_idf=False,
                                sublinear_tf=False)
        model.fit(tweets)
        token_ids = [model.transform([text]).toarray()[0] for text in tweets]
        user_feat = dict()
        user_tweet_cnt = dict()
        for i in range(len(token_ids)):
            if user_feat.get(purl[i]) is not None:
                user_feat[purl[i]] += token_ids[i]
                user_tweet_cnt[purl[i]] += 1
            else:
                user_feat[purl[i]] = token_ids[i]
                user_tweet_cnt[purl[i]] = 1

        for url in user_feat.keys():
            user_feat[url] = user_feat[url] / user_tweet_cnt[url]
        return user_feat

