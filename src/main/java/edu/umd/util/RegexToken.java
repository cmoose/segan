package edu.umd.util;

import java.util.ArrayList;

/**
 * Created by chris on 10/27/15.
 *
 */

public class RegexToken {

    protected int start = 0;
    protected int end = 0;
    protected int length = 0;
    protected String str = "X";

    public void RegexToken(int Start, int End, int Length, String Str) {
        start = Start;
        end = End;
        length = Length;
        str = Str;
    }

    public String printToken() {
        return str + ": " + Integer.toString(start) + "-" + Integer.toString(end);
    }

    //Setters
    public void end(int End) {
        this.end = End;
    }
    public void start(int Start) {
        this.start = Start;
    }
    public void length(int Length) {
        this.length = Length;
    }
    public void str(String Str) {
        this.str = Str;
    }

    //Getters
    public int end() {
        return end;
    }
    public int start() {
        return start;
    }
    public int length() {
        return length;
    }

}


