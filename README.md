# Setup
## Prerequisites
- maven
- python yaml module (to drive segan from python)

## Build
To build the latest code:

- push lass4j file to local maven repo
  - run `mvn install:install-file -Dfile=./lib/lasso4j-1.0.jar -DgroupId=edu.uci -DartifactId=lasso -Dversion=1.0 -Dpackaging=jar`
- run `mvn install`
  - This will build a `segan-1.0-SNAPSHOT.jar` in folder `target`

## segan config file
To make life easier, we've created a yaml configuration file containing all options available to preprocessing, processing, and reprocessing in segan. The file is located in `script/segan_config.yaml`. Currently, it is only used by the python driver, not segan itself.

## Preprocessing
Segan supports preprocessing your text, which builds a vocabulary file, plus counts of tokens needed for running the models. Segan requires your text to be in one of two possible formats:  
1. As a single text file  
  * where each line is a document of the format `<docid>\tab<doctext>`  
2. As a directory containing text files  
  * where each file is a document named `<docid>.txt` and the contents contain the doc text.

### Running Preprocessing
1. Configure segan_config.yaml to use the desired directories/files
2. run `python script/preprocess_segan.py`

## Processing  
Segan supports multiple models. 
1. LDA
2. SLDA
3. SNLDA
4. HDP

To run:  
* `python script/process_segan.py`  
  *(runs LDA/K=35 by default) - open the file to change these defaults

## Using the topic editor
We've built scripts to convert the outputs of LDA into a json file for viewing in distill-ery.  
To view in distill-ery:  
* `python script/editor_input_util.py` - creates the json file data.json and vocab.json from the outputs of processing

## Reprocessing the topic editor export
Once you complete making manual changes to your topics, we've created a script to create a new phis distribution needed to reprocess with segan. 
To run:  
* `python script/editor_output_util.py`
  * open this file to change the path where you stored exports.json
* `python script/reprocess_segan.py`

## The entire pipeline  
1. Prepare your raw data
2. modify script/segan_config.yaml where appropriate
3. preprocess text
4. process model(s)
5. load into distill-ery
6. export from distill-ery and reprocess through segan
7. repeat if necessary
