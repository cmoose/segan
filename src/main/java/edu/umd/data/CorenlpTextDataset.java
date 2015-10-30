package edu.umd.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import edu.stanford.nlp.util.ArraySet;
import edu.umd.util.DataUtils;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import edu.umd.util.CLIUtils;
import edu.umd.util.IOUtils;
import edu.umd.util.StatUtils;
import edu.umd.util.normalizer.ZNormalizer;

/**
 *
 * @author vietan
 */
public class CorenlpTextDataset extends TextDataset {

    protected double[] responses;
    //Stores # of total non-unique tokens per document in dataset
    private ArrayList<Integer> docsTokensCounts;
    private ArrayList<HashMap<String, Integer>> docsTokens; //lemma:count for each doc
    private ArrayList<ArrayList<String>> docsTokenids; //vocab_id:count for each doc
    private HashMap<String, Integer> wordVocabLookup;

    public CorenlpTextDataset(String name) {
        super(name);
    }

    public CorenlpTextDataset(String name, String folder) {

        super(name, folder);
    }

    public CorenlpTextDataset(String name, String folder,
                               CorpusProcessor corpProc) {
        super(name, folder, corpProc);
    }

    public double[] getResponses() {
        return this.responses;
    }

    public void setResponses(double[] responses) {
        this.responses = responses;
    }

    public void setResponses(ArrayList<Double> res) {
        this.responses = new double[res.size()];
        for (int ii = 0; ii < res.size(); ii++) {
            this.responses[ii] = res.get(ii);
        }
    }

    public double[] getResponses(ArrayList<Integer> instances) {
        double[] res = new double[instances.size()];
        for (int i = 0; i < res.length; i++) {
            int idx = instances.get(i);
            res[i] = responses[idx];
        }
        return res;
    }

    public ZNormalizer zNormalize() {
        ZNormalizer znorm = new ZNormalizer(responses);
        for (int ii = 0; ii < responses.length; ii++) {
            responses[ii] = znorm.normalize(responses[ii]);
        }
        return znorm;
    }

    public void loadResponses(String responseFilepath) throws Exception {
        if (verbose) {
            logln("--- Loading response from file " + responseFilepath);
        }

        if (this.docIdList == null) {
            throw new RuntimeException("docIdList is null. Load text data first.");
        }

        this.responses = new double[this.docIdList.size()];
        String line;
        BufferedReader reader = IOUtils.getBufferedReader(responseFilepath);
        while ((line = reader.readLine()) != null) {
            String[] sline = line.split("\t");
            String docId = sline[0];
            double docResponse = Double.parseDouble(sline[1]);
            int index = this.docIdList.indexOf(docId);

            // debug
            if (index == -1) {
                System.out.println(line);
                System.out.println(docId);
            }

            this.responses[index] = docResponse;
        }
        reader.close();
        if (verbose) {
            logln("--- --- Loaded " + this.responses.length + " responses");
        }
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + docInfoExt);
        if (verbose) {
            logln("--- Outputing document info ... " + outputFile);
        }

        BufferedWriter infoWriter = IOUtils.getBufferedWriter(outputFile);
        for (int docIndex : this.processedDocIndices) {
            infoWriter.write(this.docIdList.get(docIndex)
                    + "\t" + this.responses[docIndex]
                    + "\n");
        }
        infoWriter.close();
    }

    @Override
    public void inputDocumentInfo(File filepath) throws Exception {
        if (verbose) {
            logln("--- Reading document info from " + filepath);
        }

        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        String[] sline;
        docIdList = new ArrayList<String>();
        ArrayList<Double> responseList = new ArrayList<Double>();

        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            docIdList.add(sline[0]);
            responseList.add(Double.parseDouble(sline[1]));
        }
        reader.close();

        this.docIds = docIdList.toArray(new String[docIdList.size()]);
        this.responses = new double[responseList.size()];
        for (int i = 0; i < this.responses.length; i++) {
            this.responses[i] = responseList.get(i);
        }
    }


    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("# docs: ").append(docIds.length).append("\n");
        double max = StatUtils.max(responses);
        double min = StatUtils.min(responses);
        double mean = StatUtils.mean(responses);
        double stdv = StatUtils.standardDeviation(responses);
        int[] bins = StatUtils.bin(responses, 5);
        str.append("range: ").append(min).append(" - ").append(max).append("\n");
        str.append("mean: ").append(mean).append(". stdv: ").append(stdv).append("\n");
        for (int ii = 0; ii < bins.length; ii++) {
            str.append(ii).append("\t").append(bins[ii]).append("\n");
        }
        return str.toString();
    }

    public static CorenlpProcessor createCorenlpProcessor() {
        int unigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "u", 1);
        int bigramCountCutoff = CLIUtils.getIntegerArgument(cmd, "b", 1);
        double bigramScoreCutoff = CLIUtils.getDoubleArgument(cmd, "bs", 5.0);
        int maxVocabSize = CLIUtils.getIntegerArgument(cmd, "V", Integer.MAX_VALUE);
        int vocTermFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-tf", 1);
        int vocTermFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-tf", Integer.MAX_VALUE);
        int vocDocFreqMinCutoff = CLIUtils.getIntegerArgument(cmd, "min-df", 1);
        int vocDocFreqMaxCutoff = CLIUtils.getIntegerArgument(cmd, "max-df", Integer.MAX_VALUE);
        int docTypeCountCutoff = CLIUtils.getIntegerArgument(cmd, "min-doc-length", 1);

        boolean stopwordFilter = cmd.hasOption("s");
        boolean lemmatization = cmd.hasOption("l");
        boolean createPOSphrases = cmd.hasOption("p");

        CorenlpProcessor corpProc = new CorenlpProcessor(
                unigramCountCutoff,
                bigramCountCutoff,
                bigramScoreCutoff,
                maxVocabSize,
                vocTermFreqMinCutoff,
                vocTermFreqMaxCutoff,
                vocDocFreqMinCutoff,
                vocDocFreqMaxCutoff,
                docTypeCountCutoff,
                stopwordFilter,
                lemmatization,
                createPOSphrases);
        // If the word vocab file is given, use it. This is usually for the case
        // where training data have been processed and now test data are processed
        // using the word vocab from the training data.
        if (cmd.hasOption("word-voc-file")) {
            String wordVocFile = cmd.getOptionValue("word-voc-file");
            corpProc.loadVocab(wordVocFile);
        }
        if (verbose) {
            logln("Processing corpus with the following settings:\n"
                    + corpProc.getSettings());
        }
        return corpProc;
    }


    public static String getHelpString() {
        return "java -cp 'dist/segan.jar' " + CorenlpTextDataset.class.getName() + " -help";
    }

    public static void main(String[] args) {
        try {
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addDataDirectoryOptions();
            addOption("response-file", "Directory of the response file");

            // text processing
            addCorpusProcessorOptions();

            // cross validation
            addCrossValidationOptions();

            addOption("run-mode", "Run mode");
            options.addOption("v", false, "Verbose");
            options.addOption("d", false, "Debug");
            options.addOption("help", false, "Help");
            options.addOption("p", false, "POS Phrasing");

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            String runMode = cmd.getOptionValue("run-mode");
            switch (runMode) {
                case "process":
                    process();
                    break;
                case "load":
                    load();
                    break;
                default:
                    throw new RuntimeException("Run mode " + runMode + " is not supported");
            }

        } catch (Exception e) {
            e.printStackTrace();
            CLIUtils.printHelp(getHelpString(), options);
            throw new RuntimeException();
        }
    }

    private static CorenlpTextDataset load() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);

        CorenlpTextDataset data = new CorenlpTextDataset(datasetName, datasetFolder);
        data.setFormatFilename(formatFile);
        data.loadFormattedData(new File(data.getDatasetFolderPath(), formatFolder));
        return data;
    }

    private static void process() throws Exception {
        String datasetName = cmd.getOptionValue("dataset");
        String datasetFolder = cmd.getOptionValue("data-folder");
        String textInputData = cmd.getOptionValue("text-data");
        String formatFolder = cmd.getOptionValue("format-folder");
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        String responseFile = cmd.getOptionValue("response-file");

        CorenlpProcessor corenlpProc = createCorenlpProcessor();
        CorenlpTextDataset dataset = new CorenlpTextDataset(datasetName, datasetFolder);
        dataset.setFormatFilename(formatFile);

        // load text data
        File textPath = new File(textInputData);
        if (textPath.isFile()) {
            dataset.loadTextDataFromFile(textInputData);
        } else if (textPath.isDirectory()) {
            dataset.loadTextDataFromFolder(textInputData);
        } else {
            throw new RuntimeException(textInputData + " is neither a file nor a folder");
        }
        dataset.loadResponses(responseFile); // load response data
        dataset.setHasSentences(cmd.hasOption("sent"));

        File outputFolder_fh = new File(dataset.getDatasetFolderPath(), formatFolder);
        String outputFolder = outputFolder_fh.getAbsolutePath();

        if (verbose) {
            logln("--- Processing data ...");
        }
        IOUtils.createFolder(outputFolder);

        //Sets dataset wordVocab,
        corenlpProc.process(dataset);

        dataset.outputWordVocab(outputFolder);
        dataset.outputTextData(outputFolder);
        dataset.outputDocumentInfo(outputFolder);

    }

    @Override
    protected void outputWordVocab(String outputFolder) throws Exception {
        File wordVocFile = new File(outputFolder, formatFilename + wordVocabExt);
        if (verbose) {
            logln("--- Outputing word vocab ... " + wordVocFile.getAbsolutePath());
        }
        DataUtils.outputVocab(wordVocFile.getAbsolutePath(), wordVocab);
    }

    /**
     * Output the formatted document data.
     *
     * @param outputFolder Output folder
     * @throws java.lang.Exception
     */
    @Override
    protected void outputTextData(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + numDocDataExt);
        if (verbose) {
            logln("--- Outputing main numeric data ... " + outputFile);
        }


        BufferedWriter dataWriter = IOUtils.getBufferedWriter(outputFile);

        Boolean debugOutputTokenString = false;
        if (debugOutputTokenString) {
            for (int d = 0; d < this.docsTokens.size(); d++) {
                // write main data
                dataWriter.write(Integer.toString(this.docsTokensCounts.get(d)));
                for (String type : this.docsTokens.get(d).keySet()) {
                    dataWriter.write(" " + type + ":" + this.docsTokens.get(d).get(type));
                }
                dataWriter.write("\n");

                // save the doc id
                this.processedDocIndices.add(d);
            }
        } else {
            for (int d = 0; d < this.docsTokenids.size(); d++) {
                dataWriter.write(Integer.toString(this.docsTokensCounts.get(d)));
                for (String lemma_count : this.docsTokenids.get(d)) {
                    dataWriter.write(" " + lemma_count);
                }
                dataWriter.write("\n");

                // save the doc id
                this.processedDocIndices.add(d);
            }
        }

        dataWriter.close();
    }

    public void setWordVocab(ArrayList<String> uniqueVocab) {

        this.wordVocab = uniqueVocab;
        this.wordVocabLookup = new HashMap<>();

        //Create word vocab lookup hash map
        for (int i=0; i < uniqueVocab.size(); i++) {
            String word = uniqueVocab.get(i);

            this.wordVocabLookup.put(word, i);
        }
    }

    public void setDocsTokensCount(ArrayList<Integer> docsTokensCounts) {
        this.docsTokensCounts = docsTokensCounts;
    }

    public void setDocsTokens(ArrayList<HashMap<String, Integer>> docsTokens) {

        this.docsTokens = docsTokens;
        int D = docsTokens.size();

        ArrayList<ArrayList<String>> docsTokenIDs = new ArrayList<>();
        //Iterate through each document, store tokenID:count for each token via wordVocab lookup
        for (int d=0; d < D; d++) {

            ArrayList<String> docTokensID = new ArrayList<>();
            Set<String> docTokenKeys = docsTokens.get(d).keySet();
            for (String token : docTokenKeys) {
                Integer count = docsTokens.get(d).get(token);
                Integer tokenID = this.wordVocabLookup.get(token);
                docTokensID.add(tokenID.toString() + ":" + count.toString());
            }

            docsTokenIDs.add(docTokensID);
        }

        this.docsTokenids = docsTokenIDs;

    }

}

