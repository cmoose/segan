# Script to load the json output from the editor and create a new prior-topic-file with artificial
# probabilities to re-process with segan.
#
# Author: Chris Musialek
# Date: Nov 2015

import json
import os.path
import random
from util import *

datafolder_name = 'data'
dataset_name = 'gao'
basepath = '/Users/chris/School/UMCP/LING848-F15/final_project/{0}/{1}'.format(datafolder_name, dataset_name)
preprocessor_type = 'phrases'
preprocesspath = 'segan_preprocess/{0}'.format(preprocessor_type)
modelresultspath = 'segan_results/{0}/RANDOM_LDA_K-35_B-500_M-1000_L-50_a-0.1_b-0.1_opt-false'.format(preprocessor_type)

manual_topics = {}


def load_vocab(fh):
    vocab = []
    for word in fh:
        vocab.append(word.strip())
    return vocab


#Loads json file from editor's export
def load_export_from_editor(fh):
    raw_j = json.load(fh)
    manual_topics = {}
    for i in range(0, len(raw_j)):
        m_topic = {'keep': set(), 'stop': set(), 'label': ""}
        m_topic['label'] = raw_j[str(i)]['topiclabel']
        #TODO: need to check labels to conflate topics

        for word_obj in raw_j[str(i)]['words']:
            if word_obj['importance'] == 1:
                m_topic['keep'].add(word_obj['word'])
            elif word_obj['importance'] == -2:
                m_topic['stop'].add(word_obj['word'])

        manual_topics[i] = m_topic

    return manual_topics


#This creates the new_phis.txt file needed to rerun segan
def create_new_phis_file(fhw, new_topic_counts, vocab):
    header = str(len(new_topic_counts)) + '\n'
    fhw.write(header)

    vocab_size = str(len(vocab))

    line_prefix = vocab_size + '\t'
    for topic_counts in new_topic_counts.values():
        line = []
        for word in vocab:
            p = topic_counts[word]
            line.append(str(p))
        fhw.write(line_prefix + "\t".join(line) + '\n')


def build_new_manual_topics(new_K, manual_topics, vocab):
    new_topics = []
    all_topic_counts = {} #What we will return

    padded_K = new_K - len(manual_topics)

    def jitter():
        j = random.randint(-5, 5)
        return float(j)/100

    #Get the vocab
    vocab_size = len(vocab)
    topic_vocab_set = set(vocab)

    #Iterate over the new manually created (good) topics
    #Build their poor-man's probabilities
    #Good tokens sum to 0.75 of probability mass
    #Bad tokens (rest) sum to 0.25 of probability mass
    #Stopwords get 0 probability mass
    for _id, topic in manual_topics.items():
        topic_token_counts = Counter()
        good_tokens = topic['keep']
        stop_tokens = topic['stop']
        #Bad tokens are the remaining tokens not in good and stopwords sets
        bad_tokens = topic_vocab_set.difference(good_tokens.union(stop_tokens))

        #Skip if the topic has no marked "good" tokens
        if len(good_tokens) > 0:
            l_good = len(good_tokens)
            base_p_good = 0.75/float(l_good)

            for token in good_tokens:
                p = base_p_good + base_p_good*jitter() #new probability of token (adding a jitter of -5% to 5% of base probability value)
                topic_token_counts[token] = p

            l_bad = len(bad_tokens)
            base_p_bad = 0.25/float(l_bad)

            #Bad tokens are the rest of the vocab not in good token or stopword lists
            #These tokens' probabilities all sum to 0.25
            for token in bad_tokens:
                p = base_p_bad + base_p_bad*jitter()
                topic_token_counts[token] = p

            #For stopwords, literally set their probabilities to 0
            for token in stop_tokens:
                topic_token_counts[token] = 0

            topic_token_counts.normalize()

            all_topic_counts[_id] = topic_token_counts


    #Now, add additional topics with uniform probabilities across all vocab
    #to match total amount desired (occurs if manually annotated topics have
    #been merged)
    base_p_synthetic = 1/float(vocab_size)
    for key in range(len(all_topic_counts),len(all_topic_counts) + padded_K):
        topic_token_counts = Counter()
        for token in vocab:
            p = base_p_synthetic + base_p_synthetic*jitter()
            topic_token_counts[token] = p

        topic_token_counts.normalize()

        all_topic_counts[key] = topic_token_counts


    return all_topic_counts


def main(export_filename):
    fh_vocab = open(os.path.join(basepath, preprocesspath, '{0}.wvoc'.format(dataset_name)))
    fh_export = open(export_filename)
    #TODO: We might want to change the path here since this new file will represent a
    #different (modified) model in reality
    fhw_new_phis = open(os.path.join(basepath, modelresultspath, 'new_phis.txt'), 'wb')

    vocab = load_vocab(fh_vocab)
    new_topics_data = load_export_from_editor(fh_export)
    added_K = 4 #Number of padded topics to add
    new_topics_counts = build_new_manual_topics(len(new_topics_data) + added_K, new_topics_data, vocab)
    print "Creating new phis file for segan at {0}".format(os.path.join(basepath, modelresultspath, 'new_phis.txt'))
    create_new_phis_file(fhw_new_phis, new_topics_counts, vocab)


if __name__ == '__main__':
    #Where you saved the output of the editor
    editor_output_filename = '/Users/chris/Downloads/myexport.txt'
    main(editor_output_filename)