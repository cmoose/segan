# Script to create a pyLDAvis compatible json file from the segan preprocess
# and processing output.
#
# Author: Chris Musialek
# Date: Oct 2015

from util import *
import os.path
import pickle
import json

basepath = '/Users/chris/School/UMCP/LING848-F15/final_project/data/gao'
preprocesspath = 'segan_preprocess'
modelresultspath = 'segan_results/RANDOM_LDA_K-35_B-500_M-1000_L-50_a-0.1_b-0.1_opt-false'

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


def load_termfreq(fh):
    termfreq = Counter()
    docstokencount = []
    for line in fh:
        doctokencount = 0
        a = line.split()
        a = a[1:]
        for terms in a:
            b = terms.split(':')
            term = b[0].strip()
            count = int(b[1].strip())
            termfreq[term] += count
            doctokencount += count
        docstokencount.append(doctokencount)
    return (termfreq, docstokencount)


def convert_tf(tf):
    '''Helper function to convert Counter() object into array'''
    termfreq = []
    for i in range(0,len(tf)):
        termfreq.append(tf[str(i)])
    return termfreq


def create_pkl(phis, thetas, vocab, termfreq, docstokencount):
    fhw = open(os.path.join(basepath,modelresultspath, 'pyldavis.pkl'), 'wb')
    data = {'topic_term_dists': phis,
            'doc_topic_dists': thetas,
            'doc_lengths': docstokencount,
            'vocab': vocab,
            'term_frequency': termfreq}
    print "Creating pickle file at: {0}".format(os.path.join(basepath,modelresultspath, 'pyldavis.pkl'))
    pickle.dump(data, fhw)
    return data


def load_pkl():
    fh_pkl = open(os.path.join(basepath, modelresultspath, 'pyldavis.pkl'))
    data = pickle.load(fh_pkl)
    return data


def create_json(data):
    fhw = open(os.path.join(basepath,modelresultspath, 'pyldavis.json'), 'wb')
    print "Creating pyldavis json file at: {0}".format(os.path.join(basepath,modelresultspath, 'pyldavis.json'))
    json.dump(data, fhw)


def main():
    fh_phi = open(os.path.join(basepath, modelresultspath, 'phis.txt'))
    fh_theta = open(os.path.join(basepath, modelresultspath, 'thetas.txt'))
    fh_vocab = open(os.path.join(basepath, preprocesspath, 'gao.wvoc'))
    fh_tf = open(os.path.join(basepath, preprocesspath, 'gao.dat'))

    phis = load_phis(fh_phi)
    thetas = load_thetas(fh_theta)
    vocab = load_vocab(fh_vocab)
    (termfreq, docstokencount) = load_termfreq(fh_tf)
    data = create_pkl(phis, thetas, vocab, convert_tf(termfreq), docstokencount)
    create_json(data)


if __name__ == '__main__':
    main()


