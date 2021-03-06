# Takes raw text, runs mallet, and creates a json file suitable for distill-ery
#
# Author: Chris Musialek
# Date: Dec 2015
#

import os.path
import tempfile
import subprocess
import mallet_editor_transforms
import shutil

mallet_base_path = '/Users/chris/Downloads/mallet-2.0.8RC3/'
datasetname = 'soc'
mallet_data_path = '/Users/chris/School/UMCP/LING848-F15/pennebaker_text'
raw_text_input_fn = os.path.join(mallet_data_path, 'all_data/sed_clean') # 'text.txt' at end if a file

def run_lda(k_topics):
    #tmpdir = tempfile.mkdtemp() #Where mallet files will get written to
    tmpdir = '/var/folders/yj/t9z4w2kd7fbcykwwpk4_hhn00000gn/T/tmppmnvbN'
    print tmpdir
    mallet_filename = os.path.join(tmpdir, '{0}.mallet'.format(datasetname))

    import_type = 'import-dir' if os.path.isdir(raw_text_input_fn) else 'import-file'

    #Preprocess text into mallet format
    ppcmd = ['bin/mallet', import_type, '--input', raw_text_input_fn, '--output', mallet_filename,
           '--remove-stopwords', '--keep-sequence'] #'--preserve-case'


    #Actually run preprocessing
    pp = subprocess.Popen(" ".join(ppcmd), shell=True, cwd=mallet_base_path, stdout=subprocess.PIPE)
    for line in pp.stdout:
        print line.rstrip()

    doc_topics_fn = os.path.join(tmpdir, '{0}-doc-topics'.format(datasetname))
    word_topics_fn = os.path.join(tmpdir, '{0}-word-topics'.format(datasetname))
    topics_fn = os.path.join(tmpdir, '{0}-topics'.format(datasetname))
    word_topics_weights_fn = os.path.join(tmpdir, '{0}-word-topics-weights'.format(datasetname))
    cmd = ['bin/mallet', 'train-topics', '--input', mallet_filename, '--num-topics', str(k_topics),
           '--num-threads', '12',
           '--output-state', os.path.join(tmpdir, '{0}-state.gz'.format(datasetname)), #not really needed
           '--output-topic-keys', os.path.join(tmpdir, '{0}-topics'.format(datasetname)), #not really needed
           '--output-doc-topics', doc_topics_fn,
           '--word-topic-counts-file', word_topics_fn,
           '--topic-word-weights-file', word_topics_weights_fn]

    #Actually run processing
    #p = subprocess.Popen(" ".join(cmd), shell=True, cwd=mallet_base_path, stdout=subprocess.PIPE)
    #for line in p.stdout:
    #    print line.rstrip()

    #Build new json files for distill-ery
    mallet_editor_transforms.main(word_topics_fn, topics_fn, word_topics_weights_fn, doc_topics_fn, raw_text_input_fn, mallet_data_path)

    #shutil.rmtree(tmpdir)


if __name__ == '__main__':
    k_topic = 35
    run_lda(k_topic)
