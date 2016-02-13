# Script to create json files containing phis, thetas, vocab, and docs text for editing with the
# topic editor. Currently stores the top 100 docs and words per topic
#
# Author: Chris Musialek
# Date: Oct 2015

import os.path
import json
import heapq
import segan_config
from util import Counter
import editor_output_util
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from scipy.stats import entropy
try:
    # scikit-bio naming before 0.30
    from skbio.stats.ordination import PCoA
    skbio_old = True
except ImportError:
    # scikit-bio naming after 0.30
    from skbio.stats.ordination import pcoa
    skbio_old = False
from skbio.stats.distance import DistanceMatrix


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
        if line.strip():
            txt_data.append(unicode(line.strip(), errors='ignore'))
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


def match_labels(exports, top_topic_words, top_topic_words_ordered):
    labels = {}
    num_old_topics = 0
    num_new_topics = len(top_topic_words_ordered)
    filtered_exports = []
    for export in exports.values():
        if export['label']:
            filtered_exports.append(export)
            num_old_topics += 1
    #N*M matrix of word overlap counts and position in new topic where that occurs
    counts_matrix = np.zeros((num_old_topics,num_new_topics), dtype='(1,2)int')
    for old_topic_id, export in enumerate(filtered_exports):
        if export['label']:
            old_words = export['keep'] #list of good words tagged by user
            #For each new topic, find
            for new_topic_id, new_words in enumerate(top_topic_words_ordered):
                count = 0
                #Keep track of earliest position of a particular count to break ties later
                count_pos = 0
                for curpos, new_word in enumerate(new_words):
                    if new_word in old_words:
                        count+=1
                        count_pos = curpos
                counts_matrix[old_topic_id,new_topic_id] = [count, count_pos]
    #Now sort each label/old_topic by overlap counts, then by position in cases of a tie
    for k in range(0,num_old_topics):
        m = counts_matrix[k][:,0][:,0].max()
        best_new_topic_ids = [i for i, j in enumerate(counts_matrix[k][:,0][:,0]) if j == m]
        lowest_pos = 100
        best_new_topic_id = 0
        for tup in best_new_topic_ids:
            curpos = counts_matrix[k][:,0][tup][1]
            if curpos < lowest_pos:
                best_new_topic_id = tup
                lowest_pos = curpos
        if not labels.has_key(best_new_topic_id):
            labels[best_new_topic_id] = {filtered_exports[k]['label']: [m,lowest_pos]}
        else:
            #Means we found a new topic already labeled that has another equally good label, pick the best
            if labels[best_new_topic_id].values()[0][0] < m:
                #Current label is better, swap out
                print "Swapping label {0} for better label {1}".format(labels[best_new_topic_id].keys()[0], filtered_exports[k]['label'])
                labels[best_new_topic_id] = {filtered_exports[k]['label']: [m, lowest_pos]}
            elif labels[best_new_topic_id].values()[0][0] == m:
                if labels[best_new_topic_id].values()[0][1] > lowest_pos:
                    #Current label is better, swap out
                    print "Swapping label {0} for better label {1}".format(labels[best_new_topic_id].keys()[0], filtered_exports[k]['label'])
            else:
                print "Not assigning existing label {0}...identical to {1}".format(filtered_exports[k]['label'], labels[best_new_topic_id])
    final_labels = {}
    for k in labels.keys():
        final_labels[k] = labels[k].keys()[0]
    return final_labels


def format_top100_words(top_phis, vocab):
    out = []
    out_ordered = []
    for topic_id, topic in top_phis.items():
        top100 = set()
        top100_ordered = []
        for word_obj in topic:
            word = vocab[int(word_obj['word_id'])]
            top100.add(word)
            top100_ordered.append(word)
        out.append(top100)
        out_ordered.append(top100_ordered)
    return out, out_ordered


def _topic_coordinates(mds, topic_term_dists, topic_proportion):
   K = topic_term_dists.shape[0]
   mds_res = mds(topic_term_dists)
   assert mds_res.shape == (K, 2)
   mds_df = pd.DataFrame({'x': mds_res[:,0], 'y': mds_res[:,1], 'topics': range(1, K + 1), \
                          'Freq': topic_proportion * 100})
   return mds_df


def _jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def js_PCoA(distributions):
   """Dimension reduction via Jensen-Shannon Divergence & Principal Components

    Parameters
    ----------
    distributions : array-like, shape (`n_dists`, `k`)
        Matrix of distributions probabilities.

    Returns
    -------
    pcoa : array, shape (`n_dists`, 2)
   """
   dist_matrix = DistanceMatrix(dist.squareform(dist.pdist(distributions.values, _jensen_shannon)))
   if skbio_old:
       data = PCoA(dist_matrix).scores()
       return data.site[:,0:2]
   else:
       return pcoa(dist_matrix).samples.values[:, 0:2]


def _df_with_names(data, index_name, columns_name):
   if type(data) == pd.DataFrame:
      # we want our index to be numbered
      df = pd.DataFrame(data.values)
   else:
      df = pd.DataFrame(data)
   df.index.name = index_name
   df.columns.name = columns_name
   return df


def _series_with_name(data, name):
   if type(data) == pd.Series:
      data.name = name
      # ensures a numeric index
      return data.reset_index()[name]
   else:
      return pd.Series(data, name=name)


def _topic_coordinates(mds, topic_term_dists, topic_proportion):
   K = topic_term_dists.shape[0]
   mds_res = mds(topic_term_dists)
   assert mds_res.shape == (K, 2)
   mds_df = pd.DataFrame({'x': mds_res[:,0], 'y': mds_res[:,1], 'topics': range(1, K + 1), \
                          'Freq': topic_proportion * 100})
   return mds_df


def _topic_coordinates(mds, topic_term_dists, topic_proportion):
   K = topic_term_dists.shape[0]
   mds_res = mds(topic_term_dists)
   assert mds_res.shape == (K, 2)
   mds_df = pd.DataFrame({'x': mds_res[:,0], 'y': mds_res[:,1], 'topics': range(1, K + 1), \
                          'Freq': topic_proportion * 100})
   return mds_df


# Determines how many words are needed to reach specific probability masses (10-30%)
def explore_prob_mass(top_phis):
    prob_mass_max10 = float(0.1) #10%
    prob_mass_max20 = float(0.2) #20%
    prob_mass_max30 = float(0.3) #30%
    for topic_id, topic in top_phis.items():
        word_count = 0
        counts = {'10': 0, '20': 0, '30': 0}
        cum_prob_mass = float(0)
        for word_obj in topic:
            cum_prob_mass += word_obj['prob']
            word_count += 1
            if cum_prob_mass > prob_mass_max10 and counts['10'] is 0:
                counts['10'] = word_count
            if cum_prob_mass > prob_mass_max20 and counts['20'] is 0:
                counts['20'] = word_count
            if cum_prob_mass > prob_mass_max30 and counts['30'] is 0:
                counts['30'] = word_count
                break
        for k,count in counts.items():
            if count is 0:
                #Replace 0s with 100, so that we sort correctly
                counts[k] = 100

        print "topic {0}: {1}".format(topic_id, sorted(counts.values()))
    exit(0)


def calc_prob_mass(top_phis):
    prob_masses = {'10': [], '20': [], '30': [], '40': [], '50': []}
    for topic_id, topic in top_phis.items():
        word_count = 0
        cum_prob_mass = float(0)
        for word_obj in topic:
            cum_prob_mass += word_obj['prob']
            word_count += 1
            if word_count is 10:
                prob_masses['10'].append(cum_prob_mass)
            elif word_count is 20:
                prob_masses['20'].append(cum_prob_mass)
            elif word_count is 30:
                prob_masses['30'].append(cum_prob_mass)
            elif word_count is 40:
                prob_masses['40'].append(cum_prob_mass)
            elif word_count is 50:
                prob_masses['50'].append(cum_prob_mass)
                break
    return prob_masses


# for each visible word, return its probabilities according to each topic
# Used to map relative probability to a color in d3
def calc_norm_word_topic_probs(visible_words, phis):
    p_matrix = np.array(phis)
    word_topic_probs = {}
    for word_id in visible_words:
        word_topic_probs[word_id] = p_matrix[:,word_id].tolist()
        #word_topic_probs_normed[word_id] = p_matrix[:,word_id] / p_matrix[:,word_id].sum(axis=0)
    return word_topic_probs


# Helper function to get a set of all visible words (in top 100 of each topic)
# @return word/vocab ids
def get_visible_words(top_phis):
    visible_words = set()
    for topic in top_phis.values():
        for word_obj in topic:
            visible_words.add(word_obj['word_id'])
    return visible_words


def main():
    config = segan_config.SeganConfig()
    #Note: Only works with LDA at the moment
    if os.path.isdir(os.path.join(config.process['LDA']['modelresultspath'] + '_reprocess')):
        modelresultspath = config.process['LDA']['modelresultspath'] + '_reprocess'
    else:
        modelresultspath = config.process['LDA']['modelresultspath']

    #File handlers of raw data
    fh_phi = open(os.path.join(config.process['output_path'], modelresultspath, 'phis.txt'))
    fh_theta = open(os.path.join(config.process['output_path'], modelresultspath, 'thetas.txt'))
    fh_vocab = open(os.path.join(config.base_path, config.preprocess['output_path'], '{0}.wvoc'.format(config.dataset_name)))
    fh_tf = open(os.path.join(config.base_path, config.preprocess['output_path'], '{0}.dat'.format(config.dataset_name)))
    if os.path.isfile(os.path.join(config.base_path, config.preprocess['input_path'])):
        fh_docs_txt = open(os.path.join(config.base_path, config.preprocess['input_path']))
        txt_data = load_txt_data(fh_docs_txt)
    else:
        print "Can't find document text file, trying as directory...{0}".format(os.path.join(config.base_path, config.preprocess['input_path']))
        files = [x for x in os.listdir(os.path.join(config.base_path, config.preprocess['input_path'])) if x.endswith('txt')]
        txt_data = []
        for text_file_basename in files:
            txt_data.extend(load_txt_data(open(os.path.join(config.base_path, config.preprocess['input_path'], text_file_basename))))


    #Load raw data
    phis = load_phis(fh_phi)
    thetas = load_thetas(fh_theta)
    vocab = load_vocab(fh_vocab)
    (termfreq, docstokencount) = load_termfreq(fh_tf)
    fh_exports = None
    if os.path.isfile(config.editor['exportsfile']):
        fh_exports = open(config.editor['exportsfile'])


    #Filter data to just what we need
    num_topics = len(phis)
    top_phis = get_top100_words(phis, num_topics)
    top_thetas = get_top100_docs(thetas, num_topics)
    labels = {}
    if fh_exports:
        exports = editor_output_util.load_export_from_editor(fh_exports)
        top_topic_words, top_topic_words_ordered = format_top100_words(top_phis, vocab)
        labels = match_labels(exports, top_topic_words, top_topic_words_ordered)

    visible_words = get_visible_words(top_phis)
    word_topic_probs = calc_norm_word_topic_probs(visible_words, phis)
    prob_masses = calc_prob_mass(top_phis)

    # Load data needed for intertopic distance map
    # Taken from pyldavis python scripts
    plot_opts={'xlab': 'PC1', 'ylab': 'PC2'}
    topic_word_dists = _df_with_names(phis, 'topic', 'term')
    doc_topic_dists  = _df_with_names(thetas, 'doc', 'topic')
    doc_lengths      = _series_with_name(docstokencount, 'doc_length')
    topic_freq       = (doc_topic_dists.T * doc_lengths).T.sum()
    topic_proportion = (topic_freq / topic_freq.sum())
    topic_coordinates = _topic_coordinates(js_PCoA, topic_word_dists, topic_proportion)

    data = {'top_topic_terms': top_phis,
        'top_docs_topic': top_thetas,
        'vocab': vocab,
        'doc_txt': txt_data,
        'predicted_labels': labels,
        #needed for intertopic distance map
        'mdsDat': topic_coordinates.to_dict(orient='list'),
        'plot.opts': plot_opts,
        'prob_mass': prob_masses,
        'top_terms_probs': word_topic_probs}

    #Write data to disk
    print "Writing json to {0}".format(os.path.join(config.process['output_path'], modelresultspath))
    print "\t...vis.json"
    fhw = open(os.path.join(config.process['output_path'], modelresultspath, 'vis.json'), 'wb')
    json.dump(data, fhw)
    fhw.close()

if __name__ == '__main__':
    main()