# Script to load the json output from the editor and create a new prior-topic-file with artificial
# probabilities to re-process with segan.
#
# Author: Chris Musialek
# Date: Nov 2015

import json
import os.path
import random
from util import *
import segan_config


def load_vocab(fh):
    vocab = []
    for word in fh:
        vocab.append(word.strip())
    return vocab


#Loads json file from editor's export
def load_export_from_editor(fh):
    raw_j = json.load(fh)
    manual_topics = {}
    unique_labels = {}
    for i in range(0, len(raw_j)):
        m_topic = {'keep': set(), 'stop': set(), 'label': ""}
        label = raw_j[str(i)]['topiclabel'].strip()

        # Conflate topics if we find an identical label
        if label:
            if unique_labels.has_key(label):
                m_topic = manual_topics[unique_labels[label]]
            else:
                #Keep track of existing labels
                unique_labels[label] = i

        m_topic['label'] = raw_j[str(i)]['topiclabel']

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


def build_new_manual_topics(new_K, manual_topics, vocab, good_prob_mass_percent=0.75):
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
            base_p_good = good_prob_mass_percent/float(l_good)

            for token in good_tokens:
                p = base_p_good + base_p_good*jitter() #new probability of token (adding a jitter of -5% to 5% of base probability value)
                topic_token_counts[token] = p

            l_bad = len(bad_tokens)
            base_p_bad = (1-good_prob_mass_percent)/float(l_bad)

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


def main(export_filename, added_K, good_prob_mass_percent):
    config = segan_config.SeganConfig()

    fh_vocab = open(os.path.join(config.base_path, config.preprocess['output_path'], '{0}.wvoc'.format(config.dataset_name)))
    fh_export = open(export_filename)
    #TODO: We might want to change the path here since this new file will represent a
    #TODO: different (modified) model in reality
    #Note: Only works with LDA at the moment
    fhw_new_phis = open(config.reprocess['LDA']['prior-topic-file'], 'wb')

    vocab = load_vocab(fh_vocab)
    new_topics_data = load_export_from_editor(fh_export)
    new_topics_counts = build_new_manual_topics(len(new_topics_data) + added_K, new_topics_data, vocab, good_prob_mass_percent)
    print "Creating new phis file for segan at {0}".format(os.path.join(config.reprocess['LDA']['prior-topic-file']))
    create_new_phis_file(fhw_new_phis, new_topics_counts, vocab)


if __name__ == '__main__':
    #Where you saved the output of the editor
    editor_output_filename = '/Users/chris/Downloads/exports.json'

    #Number of padded topics to add
    added_K = 10

    #Percent of the probability mass "good/confirmed" words will take up, remainder is rest of vocab
    good_prob_mass_percent = 0.75

    main(editor_output_filename, added_K, good_prob_mass_percent)