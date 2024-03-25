package Algorithms.Wordle;

import java.util.Scanner;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

public class Wordle {

    private String wordsFileName = "words.txt";
    private String usedWordsFileName = "usedwords.txt";

    private String[] words;
    private String[] usedWords;
    private String targetWord;

    private String green = "\u001B[32m";
    private String yellow = "\u001B[33m";
    private String grayBackground = "\033[0;100m";
    private String greenBackground = "\u001B[42m";
    private String yellowBackground = "\u001B[43m";
    private String df = "\033[0m";

    private String[] alphaStrings = new String[26];

    // this is a constructor.(no parameter required)
    public Wordle() {
        this.words = loadFile(this.wordsFileName);
        this.usedWords = loadFile(this.usedWordsFileName);
        this.targetWord = setTargetWords();
        // For debugging, prints out the answer.
        System.out.println(this.targetWord + ":" + this.green
                + "(This is the answer for debugging. You can turn this off on line 39 of Wordle.java)" + this.df);

        // Create alphaStrings
        char tmpChar;
        for (int i = 65; i < this.alphaStrings.length + 65; i++) {
            tmpChar = (char) i;
            this.alphaStrings[i - 65] = tmpChar + "";
        }

    }

    // This static function validates whether the given string consists of all
    // unique letters or not.
    public static boolean allLettersUnique(String str) {
        int length = str.length();
        for (int i = 0; i < length; i++) {
            if (str.replace(str.charAt(i) + "", "").length() != length - 1) {
                return false;
            }
        }
        return true;
    }

    // This static function loads the file with the given file name.
    public static String[] loadFile(String Filename) {
        // If file is not found, it throws Exception.
        try {
            // Load all the words in words.txt into the variable arr.
            File wordsFile = new File(Filename);
            BufferedReader br = new BufferedReader(new FileReader(wordsFile));
            String str = br.readLine();
            int i = 1;
            while (str != null) {
                i++;
                str = br.readLine();
            }
            str = br.readLine();

            BufferedReader br2 = new BufferedReader(new FileReader(wordsFile));
            String[] arr = new String[i - 1];

            for (int j = 0; j < i - 1; j++) {
                str = br2.readLine();
                arr[j] = str;
            }

            br.close();
            br2.close();
            return arr;
        } catch (IOException e) {
            System.out.println(e);
            return new String[0];
        }
    }

    // This static function writes the string to the userguesses file.
    public static void writeFile(String string) {
        try {
            File file = new File("usedwords.txt");
            BufferedWriter bw = new BufferedWriter(new FileWriter(file, true));
            bw.write(string);
            bw.close();
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    // This static function returns whether the given String list contains the given
    // string.
    public static boolean checkWord(String[] strings, String string) {
        for (int i = 0; i < strings.length; i++) {
            if (strings[i] != null) {
                if (strings[i].equals(string)) {
                    return true;
                }
            }
        }
        return false;
    }

    // This method sets the class variable thargetWord to a word that has not been
    // used before.
    public String setTargetWords() {
        Random rand = new Random();
        int randi = rand.nextInt(words.length);

        boolean targetNotFound = true;
        while (targetNotFound) {
            targetNotFound = false;
            randi = rand.nextInt(words.length);
            if (checkWord(this.usedWords, this.words[randi])) {
                targetNotFound = true;
            }
        }
        return this.words[randi];
    }

    public void gameMain() {
        Scanner s = new Scanner(System.in);
        String userInput = "";
        String result = "";
        String[] userGuesses = new String[5];
        String keyboard = "";
        String letter;
        String[] lettersUsed = new String[25];

        boolean isWon = false;
        int n = 0;

        // This is the main loop that gives the user five trials.
        while (n < 5) {

            String[] lettersMatched = new String[5];
            String[] lettersUnmatchedButContained = new String[5];
            keyboard = "";
            letter = "";
            System.out.println(df + "Enter a guess:" + green);
            userInput = s.nextLine();
            if (userInput.length() != 5 || checkWord(userGuesses, userInput)) {
                if (checkWord(userGuesses, userInput)) {
                    System.out.println(this.yellow + "Invalid input (The word " + "\"" + userInput + "\""
                            + " is already typed.)" + this.df);
                } else {
                    System.out.println(
                            this.yellow + "Invalid input (The length of the string must be exactly 5) ." + this.df);
                }
            } else {
                userGuesses[n] = userInput;
                if (userInput.equals(this.targetWord)) {
                    System.out.println(this.greenBackground + this.targetWord + df + "\nYou won!\n");
                    isWon = true;
                    break;
                }
                // This part makes a string that shows the user which letters are at the correct
                // places and not.
                // The string also shows the user whether the input letters are contained in the
                // target string or not. (I followed the Wordle protocol.)
                for (int i = 0; i < userInput.length(); i++) {
                    if ((userInput.charAt(i) + "").equals(this.targetWord.charAt(i) + "")) {
                        result += this.greenBackground + userInput.charAt(i) + this.df;
                        for (int k = 0; k < 5; k++) {
                            if (lettersMatched[k] == null) {
                                lettersMatched[k] = userInput.charAt(i) + "";
                                break;
                            }
                        }
                    } else if ((this.targetWord).contains(userInput.charAt(i) + "")) {
                        result += this.yellowBackground + userInput.charAt(i) + this.df;
                        for (int k = 0; k < 5; k++) {
                            if (lettersUnmatchedButContained[k] == null) {
                                lettersUnmatchedButContained[k] = userInput.charAt(i) + "";
                                break;
                            }
                        }
                    } else {
                        result += this.grayBackground + userInput.charAt(i) + this.df;
                    }
                    lettersUsed[5 * n + i] = userInput.charAt(i) + "";
                }

                // Makes the keyboard representation for each loop (each trial).
                for (int j = 0; j < this.alphaStrings.length; j++) {

                    if (checkWord(lettersUsed, this.alphaStrings[j].toLowerCase())) {
                        letter = this.grayBackground + this.alphaStrings[j] + df;
                    } else {
                        letter = this.alphaStrings[j];
                    }

                    if (checkWord(lettersMatched, this.alphaStrings[j].toLowerCase())) {
                        letter = this.greenBackground + this.alphaStrings[j] + df;
                    } else if (checkWord(lettersUnmatchedButContained, this.alphaStrings[j].toLowerCase())) {
                        letter = this.yellowBackground + this.alphaStrings[j] + df;
                    }

                    if (this.alphaStrings[j].equals("I") || this.alphaStrings[j].equals("R")) {
                        letter += "\n";
                    }

                    keyboard += letter;
                }

                System.out.println(result);
                System.out.println("\n\n");
                result += "\n";
                n++;
                System.out.println(keyboard + "\n");
            }
        }
        if (!isWon) {
            System.out.println("You lost.\n");
        }
        writeFile(this.targetWord + "\n");
    }

    public static void main(String[] args) {

        // Reset the file usedwords.txt everytime this program starts over.
        try {
            // the case that the file already exists.
            File file = new File("usedwords.txt");
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));
            bw.write("");
            bw.close();
        } catch (IOException e) {
            // the case that the file doesn't exist in the same directory.
            try {
                Files.createFile(Paths.get("usedwords.txt"));
            } catch (IOException directoryException) {
                System.out.println(directoryException);
            }
        }

        Scanner scnr = new Scanner(System.in);
        String userInput;
        int numGames = 0;

        while (true) {
            if (numGames > 0) {
                System.out.println("Type one of the commands below.");
                System.out.println("    exit -> To quit.");
                System.out.println("    continue -> To start the game again.");

                userInput = scnr.nextLine().toLowerCase();

                if (userInput.equals("exit")) {
                    System.out.println("Game ended.");
                    break;
                } else if (userInput.equals("continue")) {
                    Wordle wordle = new Wordle();
                    wordle.gameMain();
                } else {
                    System.out.println("\u001B[33m" + "Invalid command. Type " + "\u001B[32m" + "\"continue\" "
                            + "\u001B[33m" + "or " + "\u001B[32m" + "\"exit\" " + "\033[0m");
                }

            } else {
                Wordle wordle = new Wordle();
                wordle.gameMain();
            }
            numGames++;
        }
        scnr.close();
    }
}