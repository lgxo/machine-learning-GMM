import numpy as np

if __name__ == '__main__':
    a = np.array([[8, 8, 8], [4, 4, 4]])
    print("a:")
    print(a)
    b = np.array([1, 2, 4]).reshape(1, 3)
    print("b:")
    print(b)
    # print("a/b:")
    # print(a/b)
    c = np.array([2, 1/2]).reshape(2, 1)
    print("a*c:")
    print(a*c)
