# Utility functions to convert output of mallet into form for distill-ery
#
# Author: Chris Musialek
# Date: Dec 2015
#

import os.path
import json
import heapq


#Doc-topic distributions
def load_thetas(fh):
    thetas = []
    for line in fh:
        doc = [float(x) for x in line.split()[2:]]
        thetas.append(doc)
    return thetas


def load_word_topics_weights(fh):
    probs = {}
    for line in fh:
        l = line.strip().split('\t')
        topic_id = int(l[0])
        if not probs.has_key(topic_id):
            probs[topic_id] = []
        token = l[1]
        raw_prob = float(l[2])
        probs[topic_id].append(raw_prob)
    return probs


def load_txt_data(fh):
    txt_data = []
    for line in fh:
        if line.strip():
            a = line.split('\t')
            if len(a) == 1:
                txt_data.append(unicode(a[0].strip(), errors='ignore')) #TODO: this is a hack, it will drop data
            else:
                txt_data.append(unicode(a[1].strip(), errors='ignore'))
    return txt_data


def load_vocab(fh):
    vocab = []
    for line in fh:
        l = line.split()
        vocab.append(l[1].strip())
    return vocab


def get_phis(word_topics_weights, vocab_size):
    phis = []
    for i in range(0,len(word_topics_weights)):
        word_topic_dist = [float(x/vocab_size) for x in word_topics_weights[i]]
        phis.append(word_topic_dist)
    return phis


def get_ranked_words(t, phis):
    pq = []
    for w_id in range(0, len(phis[0])):
        heapq.heappush(pq, (phis[t][w_id], w_id))

    return pq


def get_ranked_docs(topic, theta):
    pq = []
    for doc_id, doc_thetas in enumerate(theta):
        heapq.heappush(pq, (doc_thetas[topic], doc_id))
    return pq


def get_top100_words(phis, num_topics):
    words_ranked_topics = {}
    top_words_ranked_topics = {}
    for t in range(0,num_topics):
        pq = get_ranked_words(t, phis)
        words_ranked_topics[t] = pq

    for topic in range(0,num_topics):
        best = heapq.nlargest(100, words_ranked_topics[topic])
        top_words_ranked_topics[topic] = []
        for word in best:
            top_words_ranked_topics[topic].append({'prob': word[0], 'word_id': word[1]})

    return top_words_ranked_topics


def get_top100_docs(thetas, num_topics):
    topics_ranked_docs = {}
    top_topics_ranked_docs = {}
    for t in range(0,num_topics):
        pq = get_ranked_docs(t, thetas)
        topics_ranked_docs[t] = pq

    for topic in range(0,num_topics):
        best = heapq.nlargest(100, topics_ranked_docs[topic])
        top_topics_ranked_docs[topic] = []
        for doc in best:
            top_topics_ranked_docs[topic].append({'prob': doc[0], 'doc_id': doc[1]})

    return top_topics_ranked_docs


def main(word_topics_fn, word_topics_weights_fn, doc_topics_fn, raw_text_input_fn, json_output_path):
    #Load raw data
    vocab = load_vocab(open(word_topics_fn))
    probs = load_word_topics_weights(open(word_topics_weights_fn))
    phis = get_phis(probs, len(vocab))
    thetas = load_thetas(open(doc_topics_fn))
    if os.path.isdir(raw_text_input_fn):
        files = [x for x in os.listdir(raw_text_input_fn) if x.endswith('txt')]
        txt_data = []
        for single_text_input_fn in files:
            txt_data.extend(load_txt_data(open(os.path.join(raw_text_input_fn, single_text_input_fn))))
    else:
        txt_data = load_txt_data(open(raw_text_input_fn))

    #Filter data to just what we need
    num_topics = len(phis)
    top_phis = get_top100_words(phis, num_topics)
    top_thetas = get_top100_docs(thetas, num_topics)

    data = {'top_topic_terms': top_phis,
            'top_docs_topic': top_thetas,
            'vocab': vocab,
            'doc_txt': txt_data}

    #Write data to disk
    print "Writing json to {0}".format(os.path.join(json_output_path, 'vis.json'))
    fhw = open(os.path.join(json_output_path, 'vocab.json'), 'wb')
    json.dump(vocab, fhw)
    fhw = open(os.path.join(json_output_path, 'vis.json'), 'wb')
    json.dump(data, fhw)

