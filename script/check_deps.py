# Checks for all the dependencies that you need to build and run segan's preprocessing, processing
# as well as pyLDAvis
#
# Author: Chris Musialek
# Date: Oct 2015

from distutils.spawn import find_executable
import os.path


def check_cmds(exec_strings):
    has_prereqs = True

    for exec_str in exec_strings:
        executable = find_executable(exec_str)
        if executable:
            print "+ [OK] Found {0}, at location {1}...good".format(exec_str, executable)
        else:
            print "+ [ERROR] Couldn't find '{0}' executable...".format(exec_str)
            has_prereqs = False

    return has_prereqs


def check_python(modules):
    print "Checking python modules..."
    for module in modules:
        try:
            __import__(module)
            print "+ [OK] Found module {0}...good".format(module)
        except ImportError:
            print "+ [ERROR] Couldn't load module {0}".format(module)


def check_jars(addt_jars):
    print "Checking additional jars needed for some parts of segan..."
    for jar in addt_jars:
        if os.path.isfile('../lib/{0}'.format(jar)):
            print "+ [OK] Found {0}...good".format(jar)
        else:
            print "+ [ERROR] Missing jar file {0} in ../lib/".format(jar)


check_cmds(['mvn', 'java', 'python', 'ipython'])
check_python(['pyLDAvis'])
check_jars(['stanford-corenlp-3.5.1-models.jar', 'stanford-corenlp-3.5.1.jar'])
