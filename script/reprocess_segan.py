# Re-runs the segan models for you (LDA, SLDA, SNLDA) based on a manually annotated topic/word distribution
# (the prior-topic-file build with editor_output_util.py)
#
# Author: Chris Musialek
# Date: Nov 2015

import segan_config
import subprocess


def reprocess_segan():
    config = segan_config.SeganConfig()
    cmd_args = config.get_reprocess_args('LDA') #Only works with LDA at the moment

    main = cmd_args.pop('main')
    _cp = cmd_args.pop('-cp')
    _other = cmd_args.pop('_other_')
    cmd = ['java', '-cp', _cp, main, _other]

    #Get new K from prior-topic-file
    fh = open(cmd_args['--prior-topic-file'])
    new_k = int(fh.next().strip())
    cmd_args['--K'] = new_k

    for argname, argval in cmd_args.items():
        cmd.append(argname)
        cmd.append(str(argval))

    print "Running: " + " ".join(cmd)

    #Actually run
    p = subprocess.Popen(" ".join(cmd), shell=True, cwd=config.segan_bin_path, stdout=subprocess.PIPE)
    for line in p.stdout:
        print line.rstrip()

if __name__ == '__main__':
    reprocess_segan()