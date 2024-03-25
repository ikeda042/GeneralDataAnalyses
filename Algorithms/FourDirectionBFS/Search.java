package Algorithms.FourDirectionBFS;

import java.util.Random;

public class Search {
    public int[][] array;

    public Search(int arraysize) {
        array = new int[arraysize][arraysize];
    }

    public void setArray(int i, int j) {
        array[i][j] = 1;
    }

    public void dfs(int[][] array, int i, int j) {
        if (j < 0 || i < 0 || j >= array.length || i >= array[1].length || array[i][j] != 1) {
            return;
        } else {
            array[i][j] = 2;
            dfs(array, i + 1, j);
            dfs(array, i - 1, j);
            dfs(array, i, j - 1);
            dfs(array, i, j + 1);
        }
    }

    public void dfsAll(int[][] array, int i, int j) {
        if (j < 0 || i < 0 || j >= array.length || i >= array[1].length || array[i][j] != 1) {
            return;
        } else {
            array[i][j] = 2;
            dfs(array, i + 1, j);
            dfs(array, i + 1, j + 1);
            dfs(array, i - 1, j);
            dfs(array, i - 1, j - 1);
            dfs(array, i, j - 1);
            dfs(array, i + 1, j - 1);
            dfs(array, i, j + 1);
            dfs(array, i - 1, j + 1);
        }
    }

    public String toString() {
        String result = "";
        for (int i = 0; i < array.length; i++) {
            if (i != 0) {
                result += "|\n|";
            } else {
                result += "\n|";
            }
            for (int j = 0; j < array[0].length; j++) {
                if (j != array[0].length - 1) {
                    result += array[i][j] + " ";
                } else {
                    result += array[i][j];
                }

            }

        }
        return result + "|";
    }

    public static void main(String[] args) {

        int arraysize = 20;
        Search search = new Search(arraysize);
        Random rand = new Random();
        for (int i = 0; i < arraysize * arraysize / 2; i++) {
            search.setArray(rand.nextInt(arraysize), rand.nextInt(arraysize));
        }
        System.out.println(search);

        // search.dfs(search.array, 10, 10);
        // System.out.println(search);

        search.dfsAll(search.array, 10, 10);
        System.out.println(search);
    }
}
