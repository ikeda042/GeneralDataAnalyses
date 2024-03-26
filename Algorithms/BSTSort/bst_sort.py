class BST:
    def __init__(self, key, left=None, right=None) -> None:
        self.left = left
        self.right = right
        self.key = key
        pass

    def insert(self, data) -> None:
        if self.key < data:
            if self.right is None:
                self.right = BST(data)
            else:
                self.right.insert(data)
        elif self.key >= data:
            if self.left is None:
                self.left = BST(data)
            else:
                self.left.insert(data)

    def BST_out(self):
        if self.left:
            self.left.BST_out()
        print(self.key)
        if self.right:
            self.right.BST_out()

    def a(self, m):
        if self.left is not None:
            m = min(self.left.key, m)
            self.left.a(m)
        if self.right is not None:
            m = min(self.right.key, m)
            self.right.a(m)
        return m

    def inorder(self, node) -> list[int]:
        if node is None:
            return []
        else:
            return self.inorder(node.left) + [node.key] + self.inorder(node.right)


def BST_sort(array: list[int]) -> list[int]:
    out_ = BST(array[0])
    for i in array[1:]:
        out_.insert(i)
    return out_.inorder(out_)


root = BST(40)
for i in range(39, 900):
    root.insert(i)
root.BST_out()


print(BST_sort([5, 4, 3, 3, 4, 1, 4, 1, 99591348, 1]))
