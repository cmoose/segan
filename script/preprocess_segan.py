# Used to create segan response.txt and text.txt files needed for preprocessing in segan
# Drives segan java preprocessing code as well
#
# Author: Chris Musialek
# Date: Oct 2015

import os.path
import csv
import pickle
import subprocess

basepath = '/Users/chris/School/UMCP/LING848-F15/final_project'
seganpath = '/Users/chris/git/segan'
segan_output_path = 'data/gao/segan_preprocess/'

def load_csv_data():
    fh = open(os.path.join(basepath, 'data/clean/all_data.csv'))
    csvreader = csv.reader(fh)
    all_lines = []
    for line in csvreader:
        all_lines.append(line)
    return all_lines


def load_tsv_data():
    all_lines = []
    fh = open(os.path.join(basepath, 'data/clean/all_data.pkl'))
    all_lines = pickle.load(fh)
    return all_lines


def create_segan_preprocessed_files(type):
    all_data = load_tsv_data()
    fhw = open(os.path.join(basepath, segan_output_path, 'text.txt'), 'wb')
    fhw_resp = open(os.path.join(basepath, segan_output_path, 'response.txt'), 'wb')
    for doc in all_data:
        _id = doc[0]
        text = doc[5]
        if type.startswith('S'):
            exp = float(doc[6])
            if exp > 0:
                fhw.write(_id + '\t' + text + '\n')
                fhw_resp.write(_id + '\t' + str(exp) + '\n')
        else:
            fhw.write(_id + '\t' + text + '\n')
            fhw_resp.write(_id + '\n')
    print "Created text.txt and response.txt in {0} directory...".format(segan_output_path)


def run_segan_preprocessing():
    #main = 'edu.umd.data.ResponseTextDataset'
    main = 'edu.umd.data.CorenlpTextDataset'
    _cp = '-cp "{0}/target/segan-1.0-SNAPSHOT.jar:{0}/lib/*"'.format(seganpath)
    _dataset = '--dataset gao'
    _textdata = '--text-data {0}'.format(os.path.join(basepath, segan_output_path, 'text.txt'))
    _responsefile = '--response-file {0}'.format(os.path.join(basepath, segan_output_path, 'response.txt'))
    _datafolder = '--data-folder data'
    _formatfolder = '--format-folder segan_preprocess'
    _other = '--run-mode process -v -d --u 5 -s -l --bs 10 --b 5 --V 10000'
    cmd = ['java', _cp, main, _dataset, _textdata, _responsefile, _datafolder, _formatfolder, _other]
    p = subprocess.Popen(" ".join(cmd), shell=True, cwd=seganpath, stdout=subprocess.PIPE)
    print "Running: " + " ".join(cmd)
    for line in p.stdout:
        print line.rstrip()

if __name__ == '__main__':
    #Create segan preprocessing files for segan to be able to run
    if not os.path.isfile(os.path.join(basepath, 'data/gao/segan_preprocess/text.txt')):
        create_segan_preprocessed_files('SLDA')

    run_segan_preprocessing()


