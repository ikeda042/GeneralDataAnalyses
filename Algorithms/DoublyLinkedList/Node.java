package Algorithms.DoublyLinkedList;

public class Node<T extends Comparable<T>> {

    private T data;
    private Node<T> next;
    private Node<T> prev;

    public Node(T data) {
        this.data = data;
        this.next = null;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    public Node<T> getNext() {
        return next;
    }

    public void setNext(Node<T> next) {
        this.next = next;
    }

    public void setPrev(Node<T> prev) {
        this.prev = prev;
    }

    public Node<T> getPrev() {
        return this.prev;
    }

}