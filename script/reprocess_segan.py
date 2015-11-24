# Re-runs the segan models for you (LDA, SLDA, SNLDA) based on a manually annotated topic/word distribution
# (the prior-topic-file build with editor_output_util.py)
#
# Author: Chris Musialek
# Date: Nov 2015

import process_segan
import os.path

datafolder_name = 'data'
dataset_name = 'gao'
basepath = '/Users/chris/School/UMCP/LING848-F15/final_project/{0}/{1}'.format(datafolder_name, dataset_name)
preprocessor_type = 'phrases'
preprocesspath = 'segan_preprocess/{0}'.format(preprocessor_type)
modelresultspath = 'segan_results/{0}/RANDOM_LDA_K-35_B-500_M-1000_L-50_a-0.1_b-0.1_opt-false'.format(preprocessor_type)

prior_topic_file = os.path.join(basepath, modelresultspath, 'new_phis.txt')


def reprocess_segan():

    options = {'LDA': {'custom': {}}}
    #Used for re-running model with a manually modified set of topics (topic/word distributions)
    options['LDA']['custom']['prior-topic-file'] = prior_topic_file
    options['LDA']['custom']['alpha'] = '100'
    options['LDA']['custom']['beta'] = '100'

    #Get new K from prior-topic-file
    fh = open(prior_topic_file)
    k_topic = int(fh.next().strip())
    process_segan.run_all(['LDA'], [k_topic], options)

if __name__ == '__main__':
    reprocess_segan()