# Runs the segan models for you (LDA, SLDA, SNLDA)
#
# Author: Chris Musialek
# Date: Oct 2015

import os.path
import subprocess

basepath = '/Users/chris/School/UMCP/LING848-F15/final_project'
seganpath = '/Users/chris/git/segan'
datafolder_name = 'data'
dataset_name = 'gao'
preprocessor_type = 'phrases'
segan_input_path = '{0}/{1}/segan_preprocess/{2}'.format(datafolder_name, dataset_name, preprocessor_type)
segan_output_path = '{0}/{1}/segan_results/{2}'.format(datafolder_name, dataset_name, preprocessor_type)
prior_topic_file = os.path.join(basepath, segan_output_path, 'RANDOM_LDA_K-35_B-500_M-1000_L-50_a-0.1_b-0.1_opt-false/', 'new_phis.txt')

def run_all(models, k_topics):
    options = {}
    options['LDA'] = {'main': 'edu.umd.sampler.unsupervised.LDA'}
    options['LDA']['custom'] = {}
    #Used for re-running model with a manually modified set of topics (topic/word distributions)
    options['LDA']['custom']['prior-topic-file'] = prior_topic_file
    options['LDA']['custom']['alpha'] = '100'
    options['LDA']['custom']['beta'] = '100'
    options['LDA']['other'] = '--init preset -v'

    options['SLDA'] = {'main': 'edu.umd.sampler.supervised.regression.SLDA'}
    options['SLDA']['other'] = '--init random -v -d -train'

    options['SNLDA'] = {'main': 'edu.umd.sampler.supervised.regression.SNLDA'}
    options['SNLDA']['custom'] = {}
    options['SNLDA']['custom']['alphas'] = '0.1,0.1'
    options['SNLDA']['custom']['betas'] = '1.0,0.5,0.1'
    options['SNLDA']['custom']['pis'] = '0.2,0.2'
    options['SNLDA']['custom']['gammas'] = '100,10'
    options['SNLDA']['custom']['mu'] = '0.0'
    options['SNLDA']['custom']['sigmas'] = '0.01,2.5,5.0'
    options['SNLDA']['other'] = '--burnIn 50 --maxIter 100 --sampleLag 10 --report 5 --init random -v -d -train'


    _cp = '-cp "{0}/target/segan-1.0-SNAPSHOT.jar:{0}/lib/*"'.format(seganpath)
    common_opts = []
    common_opts.append('--dataset {0}'.format(dataset_name))
    common_opts.append('--word-voc-file {0}/{1}.wvoc'.format(os.path.join(basepath,segan_input_path), dataset_name))
    common_opts.append('--word-file {0}/{1}.dat'.format(os.path.join(basepath,segan_input_path), dataset_name))
    common_opts.append('--info-file {0}/{1}.docinfo'.format(os.path.join(basepath,segan_input_path), dataset_name))
    common_opts.append('--output-folder {0}'.format(os.path.join(basepath, segan_output_path)))

    for model in models:
        for k in k_topics:
            #Defines the entry point class we'll run
            main = options[model]['main']

            cmd = ['java', _cp, main]

            #Add in common options
            cmd.extend(common_opts)

            #Add K
            if model is not 'SNLDA':
                cmd.append('--K {0}'.format(k))
            else:
                cmd.append('--Ks {0},3'.format(k))

            #Add custom
            if options[model].has_key('custom'):
                custom = " ".join(["--%s %s" % (k,v) for k, v in options[model]['custom'].items() if v])
                cmd.append(custom)

            #Add remaining options
            if options[model].has_key('other'):
                cmd.append(options[model]['other'])

            print "Running: " + " ".join(cmd)

            #Actually run
            p = subprocess.Popen(" ".join(cmd), shell=True, cwd=seganpath, stdout=subprocess.PIPE)
            for line in p.stdout:
                print line.rstrip()


def run_single_lda(k_topic):
    run_all(['LDA'], [k_topic])


if __name__ == '__main__':
    #models = ['LDA', 'SLDA', 'SNLDA']
    models = ['LDA']
    k_topics = [30] #All possible topic numbers
    run_all(models, k_topics)
    #run_single_lda(35)

