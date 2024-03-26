public class BST<T extends Comparable<T>> {

    private T key;
    private BST<T> left = null;
    private BST<T> right = null;

    public BST() {
    }

    public BST(T key) {
        this.key = key;
    }

    public void insert(T val) {
        if (this.key.compareTo(val) < 0) {
            if (this.right == null) {
                this.right = new BST<T>(val);
            } else {
                this.right.insert(val);
            }
        } else {
            if (this.left == null) {
                this.left = new BST<T>(val);
            } else {
                this.left.insert(val);
            }
        }
    }

    public String inorder(BST<T> root, String array) {
        if (root == null) {
            return array;
        } else {
            return this.inorder(root.left, array) + root.key + "," + this.inorder(root.right, array);
        }
    }

    public void BSTout() {
        if (this.left != null) {
            this.left.BSTout();
        }
        System.out.println(this.key);
        if (this.right != null) {
            this.right.BSTout();
        }
    }

    public String BSTsort(T[] array) {
        BST<T> out = new BST<T>(array[0]);
        for (int i = 0; i < array.length; i++) {
            out.insert(array[i]);
        }
        return "[" + out.inorder(out, "").substring(0, array.length * 2 - 1) + "]";
    }

    public static void main(String[] args) {
        BST bst = new BST<>();
        String[] array = new String[26];
        for (int i = 0; i < 26; i++) {
            array[25 - i] = "" + (char) (97 + i);
        }

        for (int i = 0; i < array.length; i++) {
            System.out.println(array[i]);
        }
        System.out.println(bst.BSTsort(array));
    }

}
