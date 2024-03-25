package Algorithms.DoublyLinkedList;

public class LinkedList<T extends Comparable<T>> {

    private Node<T> head;
    private Node<T> tail;
    boolean isReversed = false;

    public LinkedList(Node<T> head) {

        this.head = head;
        this.tail = null;

    }

    public void add(T data) {

        Node<T> curNode = head;
        Node<T> newNode = new Node<T>(data);
        while (curNode.getNext() != null) {
            curNode = curNode.getNext();
        }
        curNode.setNext(newNode);
        newNode.setPrev(curNode);
        tail = newNode;
    }

    public String toString() {
        String result = "";
        if (!isReversed) {
            Node<T> currNode = head;
            while (currNode.getNext() != null) {
                result += currNode.getData() + "->";
                currNode = currNode.getNext();
            }
            result += currNode.getData();
        } else {
            Node<T> currNode = tail;
            while (currNode.getPrev() != null) {
                result += currNode.getData() + "->";
                currNode = currNode.getPrev();
            }
            result += currNode.getData();
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println("testing LinkedList");
    }
}