# Runs the segan models for you (LDA, SLDA, SNLDA)
#
# Author: Chris Musialek
# Date: Oct 2015

import segan_config
import subprocess


def run_all(models, k_topics, config):

    for model in models:
        cmd_args = config.get_process_args(model)
        for k in k_topics:

            #Defines the entry point class we'll run
            main = cmd_args.pop('main')
            _cp = cmd_args.pop('-cp')
            _other = cmd_args.pop('_other_')

            #Update K
            if model != 'SNLDA':
                cmd_args['--K'] = k
            else:
                cmd_args['--Ks'] = k + ",".join(cmd_args['--Ks'].split(',')[1])

            cmd = ['java', '-cp', _cp, main, _other]

            for argname, argval in cmd_args.items():
                cmd.append(argname)
                cmd.append(str(argval))

            print "Running: " + " ".join(cmd)

            #Actually run
            p = subprocess.Popen(" ".join(cmd), shell=True, cwd=config.segan_bin_path, stdout=subprocess.PIPE)
            for line in p.stdout:
                print line.rstrip()


def run_single_lda(k_topic):
    config = segan_config.SeganConfig()
    run_all(['LDA'], [k_topic], config)


if __name__ == '__main__':
    #models = ['LDA', 'SLDA', 'SNLDA']
    models = ['LDA']
    config = segan_config.SeganConfig()
    k = config.process['options']['K']
    run_all(models, [k], config)

