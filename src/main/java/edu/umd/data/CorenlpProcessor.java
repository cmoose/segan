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
    private int unigramCountCutoff; // minimum count of raw unigrams
    private int bigramCountCutoff; // minimum count of raw bigrams
    private double bigramScoreCutoff; // minimum bigram score
    private int maxVocabSize; // maximum vocab size
    // minimum term frequency (for all items in the vocab, including unigrams and bigrams)
    private int vocabTermFreqMinCutoff;
    private int vocabTermFreqMaxCutoff; // maximum term frequency
    // minimum document frequency (for all items in the vocab, including unigrams and bigrams)
    private int vocabDocFreqMinCutoff;
    private int vocabDocFreqMaxCutoff; // maximum document frequency
    private int docTypeCountCutoff;  // minumum number of types in a document
    private int minWordLength = 3; // minimum length of a word type
    private boolean filterStopwords = true; // whether stopwords are filtered
    private boolean lemmatization = false; // whether lemmatization should be performed
    private boolean createPOSphrases = false; // whether to create phrases based on POS regex
    private String stopwordFile;
    private PhraseTree customPhrases;
    private String ending = "$$";

    // tools
    private StanfordCoreNLP pipeline;
    private StopwordRemoval stopwordRemoval;
    private HashMap<String, Integer> termFreq;
    private HashMap<String, Integer> docFreq;
    private HashMap<String, Integer> leftFreq;
    private HashMap<String, Integer> rightFreq;
    private HashMap<String, Integer> bigramFreq;
    private int totalBigram;
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

    //Has no custom minimum word length, otherwise identical
    CorenlpProcessor(
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


    //Has a stopword list of words, otherwise identical
    /*public CorenlpProcessor(
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
            PhraseTree customPhrases,
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
                customPhrases,
                stopwordList);
    }*/

    public CorenlpProcessor(
            boolean filterStopwords,
            boolean lemmatization,
            boolean createPOSphrases,
            ArrayList<String> stopwordList) {

        int unigramCountCutoff = 1;
        int bigramCountCutoff = 1;
        double bigramScoreCutoff = 5.0;
        int maxVocabSize = Integer.MAX_VALUE;
        int vocTermFreqMinCutoff = 1;
        int vocTermFreqMaxCutoff = Integer.MAX_VALUE;
        int vocDocFreqMinCutoff = 1;
        int vocDocFreqMaxCutoff = Integer.MAX_VALUE;
        int docTypeCountCutoff = 1;
        int minWordLength = 3;

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

        this.stopwordRemoval = new StopwordRemoval();

        if (stopwordList != null) {
            this.stopwordRemoval.setStopwords(stopwordList);
        }

        // Create a CoreNLP pipeline
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma"); //ner to add
        // props.put("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
        // props.put("ner.applyNumericClassifiers", "false");
        this.pipeline = new StanfordCoreNLP(props);

    }

    private CorenlpProcessor(
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

        this.stopwordRemoval = new StopwordRemoval(stopwordFile);


        // Create a CoreNLP pipeline
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma"); //ner to add
        // props.put("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
        // props.put("ner.applyNumericClassifiers", "false");
        this.pipeline = new StanfordCoreNLP(props);

    }


    String getSettings() {
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

    void loadVocab(String filepath) {
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

    /*
     * Lemmatizes the custom phrase list and creates a FSM for querying
     */
    public void addCustomPhrases(ArrayList<String> input) {

        //We have custom phrases, initialize our phrasetree now (otherwise we want to keep it null)
        customPhrases = new PhraseTree();

        for (String rawPhrase : input) {
            rawPhrase = rawPhrase.replace("_", " "); //replace any underscores if user used as delimiter

            //Prepare the phrase to be annotated
            Annotation annotation = new Annotation(rawPhrase);
            // run all the selected Annotators on the phrase/document
            pipeline.annotate(annotation);

            List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);

            if (sentences != null && ! sentences.isEmpty()) {
                ArrayList<String> phraseLemmaList = new ArrayList<>();

                for (CoreMap sentence : sentences) {

                    for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                        String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
                        phraseLemmaList.add(lemma);
                    }

                }

                //Add custom ending
                phraseLemmaList.add(ending);

                //Add new phrase
                String newPhrase = String.join("_", phraseLemmaList);
                customPhrases.addPhrase(newPhrase);

            } else {
                System.out.println("Ignoring input phrase: " + rawPhrase);
            }


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

                    //custom phrase token construction
                    if (customPhrases != null) {

                        ArrayList<CoreLabel> matchingSequence = new ArrayList<>();
                        int curEndingPos = 0; //Keeps track of substring matches

                        PhraseTree curPhrases = customPhrases;

                        for (int i=0; i < sentence.get(CoreAnnotations.TokensAnnotation.class).size(); i++) {
                            CoreLabel token = sentence.get(CoreAnnotations.TokensAnnotation.class).get(i);
                            String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
                            if (curPhrases.hasChild(ending)) {
                                //Found a good ending, but let's keep extending for a possibly longer one
                                curEndingPos = matchingSequence.size();
                            }
                            if (curPhrases.hasChild(lemma)) {
                                matchingSequence.add(token);
                                curPhrases = curPhrases.getNext(lemma);
                            } else {
                                if (curEndingPos > 0) {
                                    List<CoreLabel> matchList = matchingSequence.subList(0,curEndingPos);
                                    ArrayList<String> matchStrList = new ArrayList<>();

                                    for (CoreLabel match : matchList) {
                                        matchStrList.add(match.get(CoreAnnotations.LemmaAnnotation.class));

                                    }
                                    String newPhrase = StringUtils.join(matchStrList, "_");

                                    CoreLabel newPhraseToken = new CoreLabel();

                                    newPhraseToken.setValue(newPhrase);
                                    newPhraseToken.setWord(newPhrase);
                                    newPhraseToken.setLemma(newPhrase);
                                    newPhraseToken.setTag("PHR");

                                    sentenceLemmas.add(newPhraseToken);


                                    //reset data used with tree, rollback remaining tokens
                                    curPhrases = customPhrases;
                                    int rollback = matchingSequence.subList(curEndingPos, matchingSequence.size()).size();
                                    curEndingPos = 0;
                                    i = i-rollback;
                                    matchingSequence = new ArrayList<>();


                                } else {
                                    sentenceLemmas.add(token);
                                }
                            }

                        }


                    } else {
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
