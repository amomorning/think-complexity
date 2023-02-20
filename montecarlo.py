import taichi as ti
import math
import time

ti.init(ti.gpu)

@ti.kernel
def calc(n:int) -> float:
    tot = 0.0
    for i in range(n):
        x = ti.random(float)
        y = ti.random(float)
        
        if x**2 + y**2 < 1.0:
            tot += 1.0
    return 4.0 * (tot / float(n))

def standard_deviation(arr):
    n = len(arr)
    mu = sum(arr) / n
    std_sqrt = sum([(x-mu)**2 for x in arr]) / n
    return math.sqrt(std_sqrt)


def estimate(n, trials):
    start_time = time.time()
    est = []
    for t in range(trials):
        est.append(calc(n))

    std_dev = standard_deviation(est)
    cur_est = sum(est) / trials
    time_use = time.time() - start_time
    print(f'Est. = {cur_est:.6f}, Std. = {std_dev:.5f}, N = {n}, T = {time_use:.3f}')
    return cur_est, std_dev


def main():
    precision = 0.0001
    trials = 100
    n = 1000
    std_dev = precision
    while std_dev >= precision / 2:
        cur_est, std_dev = estimate(n, trials)
        n *= 2
    return cur_est

if __name__ == '__main__':
    main()

"""
CPU results: with precision 0.001
Est. = 3.131640, Std. = 0.04585, N = 1000, T = 0.045
Est. = 3.140680, Std. = 0.03443, N = 2000, T = 0.007
Est. = 3.139800, Std. = 0.02593, N = 4000, T = 0.008
Est. = 3.142080, Std. = 0.02071, N = 8000, T = 0.011
Est. = 3.139303, Std. = 0.01408, N = 16000, T = 0.019
Est. = 3.142726, Std. = 0.00981, N = 32000, T = 0.035
Est. = 3.141542, Std. = 0.00674, N = 64000, T = 0.065
Est. = 3.142155, Std. = 0.00406, N = 128000, T = 0.118
Est. = 3.141311, Std. = 0.00332, N = 256000, T = 0.237
Est. = 3.141630, Std. = 0.00207, N = 512000, T = 0.503
Est. = 3.141632, Std. = 0.00168, N = 1024000, T = 1.032
Est. = 3.141553, Std. = 0.00121, N = 2048000, T = 2.093
Est. = 3.141634, Std. = 0.00091, N = 4096000, T = 4.288
Est. = 3.141647, Std. = 0.00064, N = 8192000, T = 8.655
Est. = 3.141616, Std. = 0.00040, N = 16384000, T = 16.972


GPU results: with precision 0.0001
Est. = 3.142880, Std. = 0.05575, N = 1000, T = 0.829
Est. = 3.145660, Std. = 0.03794, N = 2000, T = 0.775
Est. = 3.140180, Std. = 0.02974, N = 4000, T = 0.760
Est. = 3.139505, Std. = 0.01714, N = 8000, T = 0.716
Est. = 3.140293, Std. = 0.01222, N = 16000, T = 0.863
Est. = 3.142056, Std. = 0.00847, N = 32000, T = 1.081
Est. = 3.141775, Std. = 0.00614, N = 64000, T = 2.289
Est. = 3.141295, Std. = 0.00458, N = 128000, T = 2.268
Est. = 3.141650, Std. = 0.00286, N = 256000, T = 2.296
Est. = 3.140954, Std. = 0.00229, N = 512000, T = 2.294
Est. = 3.141408, Std. = 0.00159, N = 1024000, T = 2.271
Est. = 3.141377, Std. = 0.00121, N = 2048000, T = 2.276
Est. = 3.141652, Std. = 0.00079, N = 4096000, T = 2.289
Est. = 3.141407, Std. = 0.00060, N = 8192000, T = 2.303
Est. = 3.141584, Std. = 0.00040, N = 16384000, T = 2.286
Est. = 3.141596, Std. = 0.00030, N = 32768000, T = 2.427
Est. = 3.141553, Std. = 0.00022, N = 65536000, T = 2.582
Est. = 3.141599, Std. = 0.00014, N = 131072000, T = 3.008
Est. = 3.141577, Std. = 0.00011, N = 262144000, T = 4.080
Est. = 3.141594, Std. = 0.00007, N = 524288000, T = 6.492
Est. = 3.141599, Std. = 0.00005, N = 1048576000, T = 11.159
"""
