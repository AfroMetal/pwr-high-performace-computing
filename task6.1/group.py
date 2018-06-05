N = 2584
r = 144


def gcd(u, v):
    while v:
        u, v = v, u % v
    return abs(u)


def group_elements():
    for i in range(1, N):
        if gcd(i, N) == 1:
            yield i


if __name__ == "__main__":

    i = 0
    for a in group_elements():
        i += 1
        try:
            assert a ** r % N == 1
        except AssertionError:
            print(f"{a}^r != 1 mod {N}")
            exit(1)

    print(f"{i} group elements checked!")
