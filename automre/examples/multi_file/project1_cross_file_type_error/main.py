from utils import lengthen, greet


def main():
    greet("world")
    unused_local = [1, 2, 3]
    return lengthen(42)


if __name__ == "__main__":
    print(main())
