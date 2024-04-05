package Algorithms.LinkedListInplaceReverse;

public class LinkedList<T extends Comparable<T>> {
    private Node<T> head;

    public LinkedList() {
        this.head = null;
    }

    public LinkedList(Node<T> head) {
        this.head = head;
    }

    public void add(T data) {
        Node<T> currNode = head;
        if (currNode == null) {
            head = new Node<T>(data);
        } else {
            while (currNode.getNext() != null) {
                currNode = currNode.getNext();
            }
            currNode.setNext(new Node<T>(data));
        }
    }

    public String toString() {
        String result = "";
        Node<T> currNode = head;
        if (currNode == null) {
            return "Null";
        }

        while (currNode.getNext() != null) {
            result += currNode.getData() + "->";
            currNode = currNode.getNext();
        }
        return result + currNode.getData() + "->Null";
    }

    public void reverse() {
        Node<T> prevNode = null;
        Node<T> currNode = head;
        Node<T> nextNode = null;

        while (currNode != null) {
            nextNode = currNode.getNext();
            currNode.setNext(prevNode);
            prevNode = currNode;
            currNode = nextNode;
        }
        head = prevNode;
    }

    public static void main(String[] args) {
        LinkedList<Integer> l = new LinkedList<Integer>(new Node<Integer>(0));
        for (int i = 1; i < 10; i++) {
            l.add(i);
        }
        System.out.println(l);
        l.reverse();
        System.out.println(l);
    }
}
