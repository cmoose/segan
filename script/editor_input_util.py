# Script to create json files containing phis, thetas, vocab, and docs text for editing with the
# topic editor. Currently stores the top 100 docs and words per topic
#
# Author: Chris Musialek
# Date: Oct 2015

import os.path
import json
import heapq
import segan_config


def load_vocab(fh):
    vocab = []
    for word in fh:
        vocab.append(word.strip())
    return vocab


def load_phis(fh):
    phis = []
    fh.next() #first line is # of topics
    for line in fh:
        topic = [float(x) for x in line.split()[1:]]
        phis.append(topic)
    return phis


def load_thetas(fh):
    thetas = []
    fh.next() #first line is # of docs
    for line in fh:
        doc = [float(x) for x in line.split()[1:]]
        thetas.append(doc)
    return thetas


def load_txt_data(fh):
    txt_data = []
    for line in fh:
        a = line.split('\t')
        txt_data.append(a[1])
    return txt_data


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


def main():
    config = segan_config.SeganConfig()
    #Note: Only works with LDA at the moment
    modelresultspath = config.process['LDA']['modelresultspath']

    #File handlers of raw data
    fh_phi = open(os.path.join(config.process['output_path'], modelresultspath, 'phis.txt'))
    fh_theta = open(os.path.join(config.process['output_path'], modelresultspath, 'thetas.txt'))
    fh_vocab = open(os.path.join(config.base_path, config.preprocess['output_path'], '{0}.wvoc'.format(config.dataset_name)))
    if os.path.isfile(os.path.join(config.base_path, config.preprocess['input_path'], 'text.txt')):
        fh_docs_txt = open(os.path.join(config.base_path, config.preprocess['input_path'], 'text.txt'))
    else:
        print "ERROR: Can't find document text files..."

    #Load raw data
    phis = load_phis(fh_phi)
    thetas = load_thetas(fh_theta)
    vocab = load_vocab(fh_vocab)
    txt_data = load_txt_data(fh_docs_txt)
    fulldata = {'topic_term_dists': phis,
        'doc_topic_dists': thetas,
        'vocab': vocab,
        'doc_txt': txt_data}

    #Filter data to just what we need
    num_topics = len(phis)
    top_phis = get_top100_words(phis, num_topics)
    top_thetas = get_top100_docs(thetas, num_topics)

    data = {'top_topic_terms': top_phis,
            'top_docs_topic': top_thetas,
            'vocab': vocab,
            'doc_txt': txt_data}

    #Write data to disk
    print "Writing json to {0}".format(os.path.join(config.process['output_path'], modelresultspath, 'data.json'))
    fhw = open(os.path.join(config.process['output_path'], modelresultspath, 'vocab.json'), 'wb')
    json.dump(vocab, fhw)
    fhw = open(os.path.join(config.process['output_path'], modelresultspath, 'data.json'), 'wb')
    json.dump(data, fhw)

if __name__ == '__main__':
    main()