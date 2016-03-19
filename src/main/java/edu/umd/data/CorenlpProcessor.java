package edu.umd.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.List;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.ling.*;


import edu.umd.main.GlobalConstants;
import edu.umd.util.*;
import edu.umd.util.IOUtils;


/**
 * Process text data using CoreNLP
 *
 * @author Chris Musialek
 */
public class CorenlpProcessor {

    private boolean verbose = true;
    // inputs
    private int D; // number of input documents
    private Set<String> excludeFromBigrams; // set of bigrams that should not be considered
    // settings
    public int unigramCountCutoff; // minimum count of raw unigrams
    public int bigramCountCutoff; // minimum count of raw bigrams
    public double bigramScoreCutoff; // minimum bigram score
    public int maxVocabSize; // maximum vocab size
    // minimum term frequency (for all items in the vocab, including unigrams and bigrams)
    public int vocabTermFreqMinCutoff;
    public int vocabTermFreqMaxCutoff; // maximum term frequency
    // minimum document frequency (for all items in the vocab, including unigrams and bigrams)
    public int vocabDocFreqMinCutoff;
    public int vocabDocFreqMaxCutoff; // maximum document frequency 
    public int docTypeCountCutoff;  // minumum number of types in a document
    public int minWordLength = 3; // minimum length of a word type 
    public boolean filterStopwords = true; // whether stopwords are filtered
    public boolean lemmatization = false; // whether lemmatization should be performed
    public boolean createPOSphrases = false; // whether to create phrases based on POS regex
    public String stopwordFile;

    // tools
    private StopwordRemoval stopwordRemoval;
    public HashMap<String, Integer> termFreq;
    public HashMap<String, Integer> docFreq;
    protected HashMap<String, Integer> leftFreq;
    protected HashMap<String, Integer> rightFreq;
    protected HashMap<String, Integer> bigramFreq;
    protected int totalBigram;
    // output data after processing
    private ArrayList<String> vocabulary;

    private final Pattern p = Pattern.compile("\\p{Punct}");

    /*public CorenlpProcessor(CorenlpProcessor corp) {
        this(corp.unigramCountCutoff,
                corp.bigramCountCutoff,
                corp.bigramScoreCutoff,
                corp.maxVocabSize,
                corp.vocabTermFreqMinCutoff,
                corp.vocabTermFreqMaxCutoff,
                corp.vocabDocFreqMinCutoff,
                corp.vocabDocFreqMaxCutoff,
                corp.docTypeCountCutoff,
                corp.filterStopwords,
                corp.lemmatization,
                corp.createPOSphrases);
                corp.stopwordList);
    }*/

    public CorenlpProcessor(
            int unigramCountCutoff,
            int bigramCountCutoff,
            double bigramScoreCutoff,
            int maxVocabSize,
            int vocTermFreqMinCutoff,
            int vocTermFreqMaxCutoff,
            int vocDocFreqMinCutoff,
            int vocDocFreqMaxCutoff,
            int docTypeCountCutoff,
            boolean filterStopwords,
            boolean lemmatization,
            boolean createPOSphrases,
            String stopwordFile) {
        this(unigramCountCutoff,
                bigramCountCutoff,
                bigramScoreCutoff,
                maxVocabSize,
                vocTermFreqMinCutoff,
                vocTermFreqMaxCutoff,
                vocDocFreqMinCutoff,
                vocDocFreqMaxCutoff,
                docTypeCountCutoff,
                3,
                filterStopwords,
                lemmatization,
                createPOSphrases,
                stopwordFile);
    }

    public CorenlpProcessor(
            int unigramCountCutoff,
            int bigramCountCutoff,
            double bigramScoreCutoff,
            int maxVocabSize,
            int vocTermFreqMinCutoff,
            int vocTermFreqMaxCutoff,
            int vocDocFreqMinCutoff,
            int vocDocFreqMaxCutoff,
            int docTypeCountCutoff,
            boolean filterStopwords,
            boolean lemmatization,
            boolean createPOSphrases,
            ArrayList<String> stopwordList) {
        this(unigramCountCutoff,
                bigramCountCutoff,
                bigramScoreCutoff,
                maxVocabSize,
                vocTermFreqMinCutoff,
                vocTermFreqMaxCutoff,
                vocDocFreqMinCutoff,
                vocDocFreqMaxCutoff,
                docTypeCountCutoff,
                3,
                filterStopwords,
                lemmatization,
                createPOSphrases,
                stopwordList);
    }

    public CorenlpProcessor(
            int unigramCountCutoff,
            int bigramCountCutoff,
            double bigramScoreCutoff,
            int maxVocabSize,
            int vocTermFreqMinCutoff,
            int vocTermFreqMaxCutoff,
            int vocDocFreqMinCutoff,
            int vocDocFreqMaxCutoff,
            int docTypeCountCutoff,
            int minWordLength,
            boolean filterStopwords,
            boolean lemmatization,
            boolean createPOSphrases,
            ArrayList<String> stopwordList) {
        this.excludeFromBigrams = new HashSet<String>();

        // settings
        this.unigramCountCutoff = unigramCountCutoff;
        this.bigramCountCutoff = bigramCountCutoff;
        this.bigramScoreCutoff = bigramScoreCutoff;
        this.maxVocabSize = maxVocabSize;

        this.vocabTermFreqMinCutoff = vocTermFreqMinCutoff;
        this.vocabTermFreqMaxCutoff = vocTermFreqMaxCutoff;

        this.vocabDocFreqMinCutoff = vocDocFreqMinCutoff;
        this.vocabDocFreqMaxCutoff = vocDocFreqMaxCutoff;

        this.docTypeCountCutoff = docTypeCountCutoff;
        this.minWordLength = minWordLength;

        this.filterStopwords = filterStopwords;
        this.lemmatization = lemmatization;
        this.createPOSphrases = createPOSphrases;

        this.termFreq = new HashMap<String, Integer>();
        this.docFreq = new HashMap<String, Integer>();

        this.leftFreq = new HashMap<String, Integer>();
        this.rightFreq = new HashMap<String, Integer>();
        this.bigramFreq = new HashMap<String, Integer>();
        this.totalBigram = 0;

        if (stopwordFile == null) {
            this.stopwordRemoval = new StopwordRemoval();
        } else {
            this.stopwordFile = stopwordFile;
            this.stopwordRemoval = new StopwordRemoval(stopwordFile);
        }
        if (stopwordList != null) {
            this.stopwordRemoval.setStopwords(stopwordList);
        }

    }

    public CorenlpProcessor(
            int unigramCountCutoff,
            int bigramCountCutoff,
            double bigramScoreCutoff,
            int maxVocabSize,
            int vocTermFreqMinCutoff,
            int vocTermFreqMaxCutoff,
            int vocDocFreqMinCutoff,
            int vocDocFreqMaxCutoff,
            int docTypeCountCutoff,
            int minWordLength,
            boolean filterStopwords,
            boolean lemmatization,
            boolean createPOSphrases,
            String stopwordFile) {
        this.excludeFromBigrams = new HashSet<String>();

        // settings
        this.unigramCountCutoff = unigramCountCutoff;
        this.bigramCountCutoff = bigramCountCutoff;
        this.bigramScoreCutoff = bigramScoreCutoff;
        this.maxVocabSize = maxVocabSize;

        this.vocabTermFreqMinCutoff = vocTermFreqMinCutoff;
        this.vocabTermFreqMaxCutoff = vocTermFreqMaxCutoff;

        this.vocabDocFreqMinCutoff = vocDocFreqMinCutoff;
        this.vocabDocFreqMaxCutoff = vocDocFreqMaxCutoff;

        this.docTypeCountCutoff = docTypeCountCutoff;
        this.minWordLength = minWordLength;

        this.filterStopwords = filterStopwords;
        this.lemmatization = lemmatization;
        this.createPOSphrases = createPOSphrases;

        this.termFreq = new HashMap<String, Integer>();
        this.docFreq = new HashMap<String, Integer>();

        this.leftFreq = new HashMap<String, Integer>();
        this.rightFreq = new HashMap<String, Integer>();
        this.bigramFreq = new HashMap<String, Integer>();
        this.totalBigram = 0;

        if (stopwordFile == null) {
            this.stopwordRemoval = new StopwordRemoval();
        } else {
            this.stopwordFile = stopwordFile;
            this.stopwordRemoval = new StopwordRemoval(stopwordFile);
        }
        

    }


    public String getSettings() {
        StringBuilder str = new StringBuilder();
        str.append("Raw unigram min count:\t").append(unigramCountCutoff).append("\n");
        str.append("Raw bigram min count:\t").append(bigramCountCutoff).append("\n");
        str.append("Bigram min score:\t").append(bigramScoreCutoff).append("\n");
        str.append("Vocab term min freq:\t").append(vocabTermFreqMinCutoff).append("\n");
        str.append("Vocab term max freq:\t").append(vocabTermFreqMaxCutoff).append("\n");
        str.append("Vocab doc min freq:\t").append(vocabDocFreqMinCutoff).append("\n");
        str.append("Vocab doc max freq:\t").append(vocabDocFreqMaxCutoff).append("\n");
        str.append("Doc min word type:\t").append(docTypeCountCutoff).append("\n");
        str.append("Max vocab size:\t").append(maxVocabSize).append("\n");
        str.append("Word min length:\t").append(minWordLength).append("\n");
        str.append("Filter stopwords:\t").append(filterStopwords).append("\n");
        str.append("Lemmatization:\t").append(lemmatization).append("\n");
        str.append("POS phrasing:\t").append(createPOSphrases).append("\n");
        str.append("Stopword file:\t").append(stopwordFile).append("\n");
        return str.toString();
    }

    public void setVerbose(boolean v) {
        this.verbose = v;
    }

    public ArrayList<String> getVocab() {
        return this.vocabulary;
    }

    public void setVocab(ArrayList<String> voc) {
        this.vocabulary = voc;
    }

    public void loadVocab(String filepath) {
        try {
            this.vocabulary = new ArrayList<String>();
            BufferedReader reader = IOUtils.getBufferedReader(filepath);
            String line;
            while ((line = reader.readLine()) != null)
                this.vocabulary.add(line.trim());
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading vocab from " + filepath);
        }
    }



    /**
     * Process a set of documents in a dataset
     */
    public void process( CorenlpTextDataset dataset ) {
        String[] rawTexts = dataset.textList.toArray(new String[dataset.textList.size()]);

        if (rawTexts == null) {
            throw new RuntimeException("Both rawTexts and rawSentences have not "
                    + "been initialized yet");
        }


        // tokenize and normalize texts
        if (verbose) {
            System.out.println("Tokenizing and counting ...");
        }

        // Create a CoreNLP pipeline
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma"); //ner to add
        // props.put("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
        // props.put("ner.applyNumericClassifiers", "false");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        //String[] punct = {",", "''", "``", ".", ":", "!", "--", "-", "`", "'",
        //        "!!", "!!!", "!!!!", "$", "%", "&", "+", "/", ";", "?", "??", "???", "????", "#", "..."};
        String[] punct = {"CD", ".", ",", ":", "POS", "''", "``", "CC", "#", "$", "CD"}; //POS tags to filter out

        HashSet<String> pos_punct = new HashSet<String>(Arrays.asList(punct));

        D = rawTexts.length; //# of Documents in dataset/corpus
        Set<String> uniqueVocab = new HashSet<String>();
        ArrayList<HashMap<String, Integer>> docsTokens = new ArrayList<HashMap<String, Integer>>();

        //Stores total # of tokens (non-unique) for each document
        ArrayList<Integer> docsTokensCount = new ArrayList<Integer>();

        int stepsize = MiscUtils.getRoundStepSize(D, 10);
        for (int d = 0; d < D; d++) {
            if (verbose && d % stepsize == 0) {
                System.out.println("--- Tokenizing doc # " + d + " / " + D);
            }

            //Prepare the document to be annotated
            Annotation annotation = new Annotation(rawTexts[d]);
            // run all the selected Annotators on the document
            pipeline.annotate(annotation);

            //Total # of tokens (unique) in this document
            Integer numDocTokens = 0;

            //Hashmap of lemma:count per document
            HashMap<String, Integer> docTokens = new HashMap<String, Integer>();

            List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);

            if (sentences != null && ! sentences.isEmpty()) {
                for (CoreMap sentence: sentences) {

                    String simpleSentencePOS = ""; //Simplified POS of sentence for matching phrases

                    ArrayList<CoreLabel> sentenceLemmas = new ArrayList<CoreLabel>();

                    //Capture all the lemmas in the sentence, in order
                    for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                        String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);

                        String rawpos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);

                        sentenceLemmas.add(token);

                        if (createPOSphrases) {
                            //Store only the first letter of each POS for the regex to work currently
                            simpleSentencePOS = simpleSentencePOS + rawpos.substring(0,1);
                        }

                    }

                    ArrayList<CoreLabel> finalSentTokenList = new ArrayList<CoreLabel>();


                    if (createPOSphrases) {
                        sentenceLemmas = RegexMatcher.runRegexMatcher(simpleSentencePOS, sentenceLemmas);
                    }


                    //Remove stopwords
                    for (CoreLabel token : sentenceLemmas) {
                        String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
                        String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                        if (filterStopwords) {
                            if (!stopwordRemoval.isStopword(lemma.toLowerCase()) && !pos_punct.contains(pos)) {
                                finalSentTokenList.add(token);
                                MiscUtils.incrementMap(docTokens, lemma);

                            }
                        } else {
                            if (!pos_punct.contains(pos)) {
                                finalSentTokenList.add(token);
                                MiscUtils.incrementMap(docTokens, lemma);
                            }
                        }


                    }



                    //Add # of sent tokens to doc list
                    numDocTokens = docTokens.size();

                    for (CoreLabel token : finalSentTokenList) {
                        uniqueVocab.add(token.lemma());
                    }

                }
            }

            docsTokensCount.add(d, numDocTokens);
            docsTokens.add(d, docTokens);


        }

        //Now, sort vocab, then iterate through one more time
        ArrayList<String> sortedVocab = new ArrayList<String>(uniqueVocab);
        Collections.sort(sortedVocab);


        dataset.setWordVocab(sortedVocab);
        dataset.setDocsTokensCount(docsTokensCount);
        dataset.setDocsTokens(docsTokens);


    }


}
