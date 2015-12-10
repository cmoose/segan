# Class to encapsulate all configurations needed to run segan
# preprocessing, processing, reprocessing, and all other interactions
# with segan.
# These options are intentionally explicit to help with understanding
# the options being passed to segan.
#
# Author: Chris Musialek
# Date: Dec 2015
#
import os.path
import yaml


class SeganConfig:
    def __init__(self, yaml_file='segan_config.yaml'):
        scriptpath = os.path.dirname(os.path.realpath(__file__))
        if os.path.isfile(os.path.join(scriptpath, yaml_file)):
            yml_cnf = yaml.load(open(os.path.join(scriptpath, yaml_file)))

            self.base_path = yml_cnf['globals']['basepath']
            self.segan_bin_path = yml_cnf['globals']['seganbinpath']
            self.dataset_name = yml_cnf['globals']['datasetname']
            self.javaclasspath = '{0}/target/segan-1.0-SNAPSHOT.jar:{0}/target/lib/*'.format(self.segan_bin_path)

            #preprocessing
            self.preprocess = {}
            self.preprocess['main'] = yml_cnf['preprocess']['main']
            self.preprocess['input_path'] = os.path.join(self.base_path, yml_cnf['preprocess']['inputfolderpath'])
            self.preprocess['output_path'] = os.path.join(self.base_path, yml_cnf['preprocess']['outputfolderpath'])
            self.preprocess['response-file'] = yml_cnf['preprocess']['responsefilepath']
            self.preprocess['stopword_filepath'] = yml_cnf['preprocess']['stopwordfilepath']
            self.preprocess['phrases'] = yml_cnf['preprocess']['phrases']
            self.preprocess['other'] = yml_cnf['preprocess']['other']


            #Options for processing (running models)
            self.process = {}
            self.process['input_path'] = os.path.join(self.base_path, yml_cnf['process']['inputfolderpath'])
            self.process['output_path'] = os.path.join(self.base_path, yml_cnf['process']['outputfolderpath'])
            self.process['v_d'] = yml_cnf['process']['verbosedebug']

            self.process['options'] = {}
            self.process['options']['model'] = yml_cnf['process']['options']['defaultmodel']
            self.process['options']['K'] = yml_cnf['process']['options']['defaultK']

            #Entry point class names
            self.process['main'] = {}
            self.process['main']['LDA'] = yml_cnf['process']['LDA']['main']
            self.process['main']['SLDA'] = yml_cnf['process']['SLDA']['main']
            self.process['main']['SNLDA'] = yml_cnf['process']['SNLDA']['main']

            #Specific model options
            self.process['LDA'] = {}
            self.process['LDA']['alpha'] = yml_cnf['process']['LDA']['custom']['alpha']
            self.process['LDA']['beta'] = yml_cnf['process']['LDA']['custom']['beta']
            self.process['LDA']['init'] = yml_cnf['process']['LDA']['custom']['init']
            self.process['LDA']['burnIn'] = yml_cnf['process']['LDA']['custom']['burnIn']
            self.process['LDA']['maxIter'] = yml_cnf['process']['LDA']['custom']['maxIter']
            self.process['LDA']['sampleLag'] = yml_cnf['process']['LDA']['custom']['sampleLag']
            self.process['LDA']['other'] = yml_cnf['process']['LDA']['other']
            self.process['LDA']['modelresultspath'] = self.build_model_results_path('LDA')

            self.process['SLDA'] = {}
            self.process['SLDA']['alpha'] = yml_cnf['process']['SLDA']['custom']['alpha']
            self.process['SLDA']['beta'] = yml_cnf['process']['SLDA']['custom']['beta']
            self.process['SLDA']['init'] = yml_cnf['process']['SLDA']['custom']['init']
            self.process['SLDA']['burnIn'] = yml_cnf['process']['SLDA']['custom']['burnIn']
            self.process['SLDA']['maxIter'] = yml_cnf['process']['SLDA']['custom']['maxIter']
            self.process['SLDA']['sampleLag'] = yml_cnf['process']['SLDA']['custom']['sampleLag']
            self.process['SLDA']['other'] = yml_cnf['process']['SLDA']['other']
            #self.process['SLDA']['modelresultspath'] = self.build_model_results_path('SLDA')

            self.process['SNLDA'] = {}
            self.process['SNLDA']['alphas'] = yml_cnf['process']['SNLDA']['custom']['alphas']
            self.process['SNLDA']['betas'] = yml_cnf['process']['SNLDA']['custom']['betas']
            self.process['SNLDA']['pis'] = yml_cnf['process']['SNLDA']['custom']['pis']
            self.process['SNLDA']['gammas'] = yml_cnf['process']['SNLDA']['custom']['gammas']
            self.process['SNLDA']['mu'] = yml_cnf['process']['SNLDA']['custom']['mu']
            self.process['SNLDA']['sigmas'] = yml_cnf['process']['SNLDA']['custom']['sigmas']
            self.process['SNLDA']['sampleLag'] = yml_cnf['process']['SNLDA']['custom']['sampleLag']
            self.process['SNLDA']['other'] = yml_cnf['process']['SNLDA']['other']
            #self.process['SNLDA']['modelresultspath'] = self.build_model_results_path('SNLDA')


            #Options for Re-processing (using a custom theta distribution)
            #Only works with LDA at the moment
            self.reprocess = {}
            self.reprocess['LDA'] = {}
            if not os.path.isfile(yml_cnf['reprocess']['LDA']['custom']['priortopicfile']):
                self.reprocess['LDA']['prior-topic-file'] = \
                    os.path.join(self.process['output_path'],
                                 self.process['LDA']['modelresultspath'],
                                 yml_cnf['reprocess']['LDA']['custom']['priortopicfile'])
            else:
                self.reprocess['LDA']['prior-topic-file'] = yml_cnf['reprocess']['LDA']['custom']['priortopicfile']

    def argmap(self, arg):
        d = {'burnIn': 'B', 'maxIter': 'M', 'sampleLag': 'L', 'alpha': 'a', 'beta': 'b'}
        return d[arg]

    #Example: RANDOM_LDA_K-35_B-500_M-1000_L-50_a-0.1_b-0.1
    def build_model_results_path(self, model):
        opt = self.process[model]
        K_part = 'K-{0}'.format(self.process['options']['K'])
        path = [opt['init'].upper(), model, K_part]
        for key in ['burnIn', 'maxIter', 'sampleLag', 'alpha', 'beta']:
            path.append('{0}-{1}'.format(self.argmap(key), opt[key]))

        return "_".join(path)


    #Actually builds the command arguments for segan to run
    def get_process_args(self, model=''):
        if model == '':
            model = self.process['options']['model']

        cmd = {}
        cmd['-cp'] = self.javaclasspath
        cmd['main'] = self.process['main'][model]
        cmd['--dataset'] = self.dataset_name
        cmd['--word-voc-file'] = os.path.join(self.process['input_path'], '{0}.wvoc'.format(self.dataset_name))
        cmd['--info-file'] = os.path.join(self.process['input_path'], '{0}.docinfo'.format(self.dataset_name))
        cmd['--word-file'] = os.path.join(self.process['input_path'], '{0}.dat'.format(self.dataset_name))
        cmd['--output-folder'] = os.path.join(self.process['output_path'])

        if model is not 'SNLDA':
            cmd['--K'] = self.process['options']['K']
        else:
            cmd['--Ks'] = '{0},3'.format(self.process['options']['K'])

        for key in [x for x in self.process[model].keys() if (x != 'other' and x != 'modelresultspath')]:
            if self.process[model][key] is not None:
                cmd['--'+key] = self.process[model][key]

        cmd['_other_'] = self.process['v_d']

        if self.process[model]['other']:
            cmd['_other_'] += " " + self.process[model]['other']

        return cmd


    def get_preprocess_args(self):
        cmd = {}
        cmd['-cp'] = self.javaclasspath
        cmd['main'] = self.preprocess['main']
        cmd['--dataset-name'] = self.dataset_name
        if os.path.isfile(os.path.join(self.base_path, self.preprocess['input_path'], 'text.txt')):
            cmd['--input-text-data'] = os.path.join(self.base_path, self.preprocess['input_path'], 'text.txt')
        else:
            cmd['--input-text-data'] = os.path.join(self.base_path, self.preprocess['input_path'])
        cmd['--output-folder'] = os.path.join(self.base_path, self.preprocess['output_path'])
        cmd['_other_'] = self.preprocess['other']

        if self.preprocess['stopword_filepath'] is not None:
            if os.path.isfile(os.path.join(self.base_path, self.preprocess['stopword_filepath'])):
                cmd['--stopword-file'] = os.path.join(self.base_path, self.preprocess['stopword_filepath'])

        if self.preprocess['response-file'] is not None:
            cmd['--response-file'] = self.preprocess['response-file']

        if self.preprocess['phrases']:
            cmd['_other_'] += ' -p'

        return cmd


    def get_reprocess_args(self, model):
        cmd = self.get_process_args(model)
        cmd['--prior-topic-file'] = self.reprocess[model]['prior-topic-file']

        return cmd


