# Used to create segan response.txt and text.txt files needed for preprocessing in segan
# Drives segan java preprocessing code as well
#
# Author: Chris Musialek
# Date: Oct 2015

import subprocess
import segan_config


def run_segan_preprocessing():
    config = segan_config.SeganConfig()
    cmd_args = config.get_preprocess_args()

    _main = cmd_args.pop('main')
    _cp = cmd_args.pop('-cp')
    _other = cmd_args.pop('_other_')

    cmd = ['java', '-cp', _cp, _main, _other]

    for argname, argval in cmd_args.items():
        cmd.append(argname)
        cmd.append(argval)

    print "Running: " + " ".join(cmd)

    #Actually run
    p = subprocess.Popen(" ".join(cmd), shell=True, cwd=config.segan_bin_path, stdout=subprocess.PIPE)

    for line in p.stdout:
        print line.rstrip()


if __name__ == '__main__':
    run_segan_preprocessing()


