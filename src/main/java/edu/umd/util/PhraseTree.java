package edu.umd.util;

import java.util.HashMap;
import java.util.Arrays;
import org.apache.commons.lang3.StringUtils;

/**
 * Custom FSM/tree for querying the existence of a custom phrase
 */
public class PhraseTree {
    private String data;
    private PhraseTree parent;
    private HashMap<String, PhraseTree> children;

    public PhraseTree() {
        this.data = "root";
        this.children = new HashMap<>();

    }

    public PhraseTree(String key, String data) {
        this.data = key;
        this.children = new HashMap<>();

        if (data != "") {
            this.addPhrase(data);
        }


    }

    public void addPhrase(String phrase) {
        String[] tokens = phrase.split("_");

        if (children.containsKey(tokens[0])) {
            PhraseTree subPhrase = this.children.get(tokens[0]);
            subPhrase.addPhrase(StringUtils.join(Arrays.copyOfRange(tokens, 1,tokens.length), "_"));
        } else {
            this.children.put(tokens[0], new PhraseTree(tokens[0], StringUtils.join(Arrays.copyOfRange(tokens, 1,tokens.length), "_")));
        }
    }


    public Boolean hasChild(String key) {
        return this.children.containsKey(key);
    }

    public PhraseTree getNext(String key) {
        return this.children.get(key);
    }

    public PhraseTree addChild(String key) {
        PhraseTree childNode = new PhraseTree();
        this.children.put(key, new PhraseTree());
        return childNode;
    }
}



