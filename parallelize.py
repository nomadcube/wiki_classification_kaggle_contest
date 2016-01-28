from multiprocessing import Pool
from time import time


def f(x):
    return x * x


if __name__ == '__main__':
    mp_start_time = time()
    r = range(100000)
    p = Pool(1)
    p.map(f, r)
    print(time() - mp_start_time)

    loop_start_time = time()
    for i in r:
        f(i)
    print(time() - loop_start_time)
