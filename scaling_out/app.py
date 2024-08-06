from beam import function


@function(cpu=0.1)
def square(i: int):
    return i**2


def main():
    numbers = list(range(10))
    squared = []

    # Run a remote container for every item in list
    for result in square.map(numbers):
        print(result)
        squared.append(result)

    print("result", squared)


if __name__ == "__main__":
    main()
