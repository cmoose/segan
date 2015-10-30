package edu.umd.data;

import java.lang.reflect.Array;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.CoreLabel;
import edu.umd.util.*;

/**
 * Created by Chris Musialek on 10/27/15.
 */
public class RegexMatcher {

    public static final String regexPattern = "((J|N)+|((J|N)*(NI)?)(J|N)*)N";
    /*
     * Runs regex on a single CoreNLP sentence of annotations
     */
    public static ArrayList<CoreLabel> runRegexMatcher(String simpleSentencePOS, ArrayList<CoreLabel> sentLemmas) {

        //Using regex, returns a list of phrase tokens start and end position in the sentence.
        ArrayList<ArrayList<Integer>> sentPhrasePositions = RegexMatcher.getAllMatches(simpleSentencePOS, regexPattern);


        HashMap<CoreLabel, ArrayList<CoreLabel>> tokensToRemove = new HashMap<>();
        ArrayList<Integer> tokensToRemoveIndex = new ArrayList<>();


        //Make phrase tokens and capture unigrams to remove for the entire sentence
        for (ArrayList<Integer> phrasePosition : sentPhrasePositions) {
            ArrayList<CoreLabel> newPhraseList = new ArrayList<>();
            Integer start = phrasePosition.get(0);
            Integer end = phrasePosition.get(1);
            for (int i = start; i < end; i++) {
                newPhraseList.add(sentLemmas.get(i));
            }

            //Construct a new Token object using phrase
            CoreLabel newPhraseToken = new CoreLabel();

            String newPhrase = "";
            for (CoreLabel t : newPhraseList) {
                String l = t.lemma();
                if (newPhrase == "") {
                    newPhrase += l;
                } else {
                    newPhrase += "_" + l;
                }
            }

            newPhraseToken.setValue(newPhrase);
            newPhraseToken.setWord(newPhrase);
            newPhraseToken.setLemma(newPhrase);
            newPhraseToken.setTag("PHR");


            //Prepare unigram tokens forming new phrase to be removed.
            //Can't actually remove until finished iterating through sentence
            //because we use the index values in sentLemmas to get the tokens

            //Get the first token, we need its index in sentLemmas.
            CoreLabel token = newPhraseList.get(0);
            Integer i = sentLemmas.indexOf(token);
            tokensToRemove.put(newPhraseToken, newPhraseList);
            tokensToRemoveIndex.add(i);

        }


        //Finally, swap out phrase tokens, and remove additional unigram tokens
        for (CoreLabel phraseToken : tokensToRemove.keySet()) {

            ArrayList<CoreLabel> unigramTokens;
            unigramTokens = tokensToRemove.get(phraseToken);
            Boolean isFirst = true;
            for (CoreLabel token : unigramTokens) {
                if (isFirst) {
                    Integer index = sentLemmas.indexOf(token);
                    sentLemmas.add(index, phraseToken);
                    sentLemmas.remove(token);
                    isFirst = false;
                } else {
                    sentLemmas.remove(token);
                }

            }


        }


        return sentLemmas;

    }

    /*
     * This is probably not necessary, plus it's very expensive on longer sequences
     */
    public static ArrayList<String> getAllMatchesExhaustive(String text, String regex) {
        ArrayList<String> matches = new ArrayList<String>();
        for (int length = 1; length <= text.length(); length++) {
            for (int index = 0; index <= text.length() - length; index++) {
                String sub = text.substring(index, index + length);
                if (sub.matches(regex)) {
                    matches.add(sub);
                }
            }
        }
        return matches;
    }

    public static ArrayList<ArrayList<Integer>> getAllMatches(String text, String regex) {

        ArrayList<RegexToken> prunedMatches = reallyGetAllMatches(text, regex);
        ArrayList<ArrayList<Integer>> prunedTokenPositions = new ArrayList<ArrayList<Integer>>();

        for (RegexToken multiToken : prunedMatches ) {
            ArrayList<Integer> startEndPos = new ArrayList<Integer>();
            startEndPos.add(multiToken.start());
            startEndPos.add(multiToken.end());
            prunedTokenPositions.add(startEndPos);
        }

        return prunedTokenPositions;
    }
    /*
     * Given a sentence of POS tags (encoded as strings) and a regex,
     * find all non-overlapping matches
     */
    public static ArrayList<RegexToken> reallyGetAllMatches(String text, String regex) {

        ArrayList<RegexToken> allMatches = new ArrayList<RegexToken>();
        Matcher m = Pattern.compile("(?=(" + regex + "))").matcher(text);
        while (m.find()) {

            RegexToken curtoken = new RegexToken();
            curtoken.start(m.start(1));
            curtoken.end(m.end(1));
            curtoken.length(m.end(1) - m.start(1));
            curtoken.str(m.group(1));

            allMatches.add(curtoken);
        }

        //Sort array by token's length so that we bias these.
        //This heuristic may need to further reviewed.
        Collections.sort(allMatches, new TokenLengthComparator());

        ArrayList<RegexToken> prunedMatches = pruneMatches(allMatches);

        return prunedMatches;

    }

    /*
     * Given a set of regex valid matched sequences, insures matches don't
     * overlap with each other
     * Biased towards picking the longest matched sequences first
     */
    public static ArrayList<RegexToken> pruneMatches(ArrayList<RegexToken> sortedMatches) {
        ArrayList<RegexToken> prunedList = new ArrayList<RegexToken>();

        Iterator<RegexToken> it = sortedMatches.iterator();
        while (it.hasNext()) {
            RegexToken cursorted = sortedMatches.remove(0);

            if (prunedList.size() == 0) {
                //No conflict possible, so just add to pruned list
                if (cursorted.length() > 1) {
                    prunedList.add(cursorted);
                }

            } else {
                //Possible conflict, iterate over existing sequences
                Iterator<RegexToken> prIt = prunedList.iterator();

                Boolean conflict = false;
                while (prIt.hasNext() && !conflict) {
                    RegexToken curpruned = prIt.next();

                    //Have three cases
                    //sequence overlaps on right side, or past
                    if (cursorted.end() > curpruned.end()) {
                        if (cursorted.start() > curpruned.end()) {
                            //No conflict yet

                        } else if (cursorted.start() > curpruned.start()) {
                            //Conflict, sequence overlaps on right side
                            conflict = true;
                        }
                        //sequence is contained within
                    } else if (cursorted.start() > curpruned.start()) {
                        //Conflict
                        conflict = true;
                        //sequence overlaps on left side, or before
                    } else if (cursorted.end() > curpruned.start()) {
                        //Conflict
                        conflict = true;
                    } else {
                        //No conflict yet
                    }
                }
                if (!conflict) {
                    if (cursorted.length() > 1) {
                        prunedList.add(cursorted);
                    }

                }
            }
        }

        return prunedList;
    }

    /*
     * TODO: Modify the regex instead to accept true Penn Tree bank POS tag
     */
    public static ArrayList<String> simplifyPOS( ArrayList<String> sentencePOS ) {

        ArrayList<String> simpleSentencePOS = new ArrayList<String>();
        for ( String POS : sentencePOS ) {
            //Note: Our current regex only cares about Nouns, Prepositions, and adjectives
            simpleSentencePOS.add(POS.substring(0, 1));
        }

        return simpleSentencePOS;
    }


    public static void main( String[] args )
    {
        ArrayList<String> tests = new ArrayList<String>();
        tests.addAll(Arrays.asList("AAA", "AAN", "NNN", "NNPNN", "NNASNPAN", "NNNPAN", "NN", "NPAN", "N"));
        for (String test : tests) {
            System.out.println("String is: " + test);
            ArrayList<RegexToken> bestMatches = reallyGetAllMatches(test, "((J|N)+|((J|N)*(NI)?)(J|N)*)N");
            for (RegexToken match : bestMatches) {
                System.out.println("\t" + match.printToken());
            }
        }

    }
}

class TokenLengthComparator implements Comparator<RegexToken> {

    public int compare(RegexToken t1, RegexToken t2) {
        return t2.length() - t1.length();
    }
}

class TokenStartComparator implements Comparator<RegexToken> {

    public int compare(RegexToken t1, RegexToken t2) {
        return t1.start() - t2.start();
    }
}
