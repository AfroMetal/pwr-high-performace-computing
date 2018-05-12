#include <iostream>
#include <omp.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/vector.h>
#include <assert.h>
#include <functional>
#include <string>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <map>
#define DIST_BITS 8
#define WILD_KANGAROO false
#define TAME_KANGAROO true
#define WILD_THREADS 2
#define TAME_THREADS 3
#define NUM_THREADS 5

using namespace std;
using namespace NTL;

string ZZToString(const ZZ &z) {
    stringstream buffer;
    buffer << z;
    return buffer.str();
}

unsigned long MaxJumps(ZZ beta) {
    unsigned long r = 1;
    ZZ res;
    res = r;

    do {
        /* res = (2^r - 1) / r */
        power(res, ZZ(2), r);
        res -= 1;
        res /= r;

        ++r;
    } while (res < beta);

    return r - 2;
}

ZZ pollardLambda(ZZ alpha, ZZ beta, ZZ P, ZZ a, ZZ b) {
    int tid;
    Vec<ZZ> dists;
    Vec<ZZ> jumps;
    unsigned long r;
    int index;
    string str;
    ZZ Q = (P - ZZ(1)) / ZZ(2);
    ZZ res, beta_min;
    bool quit = false;
    map<ZZ, tuple<ZZ, bool, int>> distinguishedValues;

    // beta_min = NUM_THREADS * sqrt(b - a) / 4
    beta_min = MulMod(ZZ(NUM_THREADS), SqrRoot(b - a), P) / ZZ(4);

    // r - max jumps
    r = MaxJumps(beta_min);

//    cout << "max jumps: " << r << endl;

    dists.SetLength(r);
    jumps.SetLength(r);

    for (int i = 0; i < r; ++i)
    {
        dists[i] = MulMod(PowerMod(ZZ(2), ZZ(i), Q), ZZ(WILD_THREADS * TAME_THREADS), Q);
        jumps[i] = PowerMod(alpha, dists[i], P);
    }

    bool kangarooType;
    ZZ dist, pos, x, step;

    #pragma omp parallel \
    num_threads(NUM_THREADS) \
    shared(res, quit, distinguishedValues) \
    private(tid, dist, pos, kangarooType, x, step, index, str) \
    firstprivate(a, b, alpha, beta, P, Q, jumps, dists, r)
    {
        tid = omp_get_thread_num();

        #pragma omp critical
        {
            dist = 0;

            if (tid < TAME_THREADS) {
                kangarooType = TAME_KANGAROO;
                ZZ i;
                i = tid;
                // dist = i*vi
//                dist = MulMod(i, vi, Q);
                // pos = g^((a + b) / 2 + iv)
                pos = PowerMod(alpha, ((a + b) / ZZ(2)) + i*ZZ(WILD_THREADS), P);
            } else {
                kangarooType = WILD_KANGAROO;
                ZZ j;
                j = tid - TAME_THREADS;
                // dist = j*vi
//                dist = MulMod(j, vi, Q);
                // pos = h * g^ju
                pos = PowerMod(alpha, MulMod(j, ZZ(TAME_THREADS), P), P);
                pos = MulMod(beta, pos, P);
            }

//            cout << "start pos:" << tid << ": " << pos << endl;
        }

        #pragma omp barrier

        do {
            str = ZZToString(pos);
            index = (int)(hash<string>{}(str) % r);

            pos = MulMod(pos, jumps[index], P);
            dist += dists[index];


            if ((pos & ZZ((1 << DIST_BITS) - 1)) == 0) {
                #pragma omp critical
                if (!quit) {
//                    cout << "distinguished:" << tid << ": " << pos << endl;
                    pair<map<ZZ, tuple<ZZ, bool, int>>::iterator, bool> ret = distinguishedValues.insert(
                            pair<ZZ, tuple<ZZ, bool, int>>(pos, make_tuple(dist, kangarooType, tid)));
                    if (!ret.second) {
                        quit = true;
//                        cout << "collision:" << get<2>(ret.first->second) << ": " << ret.first->first << endl;
                        // x = (a + b) / 2 + iv - ju + dist_TAME - dist_WILD
                        x = ZZ(a + b) / ZZ(2);
                        ZZ i, j, dist_tame, dist_wild;
                        if (kangarooType == TAME_KANGAROO) {
                            i = tid;
                            j = get<2>(ret.first->second) - TAME_THREADS;
                            dist_tame = dist % Q;
                            dist_wild = get<0>(ret.first->second) % Q;
                        } else {
                            i = get<2>(ret.first->second);
                            j = tid - TAME_THREADS;
                            dist_tame = get<0>(ret.first->second) % Q;
                            dist_wild = dist % Q;
                        }
                        AddMod(x, x, dist_tame, Q);
                        SubMod(x, x, dist_wild, Q);
                        AddMod(x, x, i * WILD_THREADS, Q);
                        SubMod(x, x, j * TAME_THREADS, Q);

                        res = x;
                    }
                }
            }
        } while (!quit);
    }

    return res;
}

int main(int argc, char **argv) {

    ZZ Q, P;
    ZZ alpha;
    ZZ beta;
    ZZ a, b;
    ZZ p, e;
    Vec<ZZ> primes;
    Vec<ZZ> exponents;
    int k, i;

    if (argc >= 4) {
        int arg = 1;

        P = conv<ZZ>(argv[arg++]);
        alpha = conv<ZZ>(argv[arg++]);
        beta = conv<ZZ>(argv[arg++]);

        k = argc - 4 >> 1;
        primes.SetLength(k);
        exponents.SetLength(k);

        for (i=0; i<k; ++i) {
            primes[i] = conv<ZZ>(argv[arg++]);
            exponents[i] = conv<ZZ>(argv[arg++]);
        }
    } else {
         cout << "Not enough arguments provided.\n"
                "You have to provide following commandline arguments: hpc_pohlig P alpha beta (p e)...\n";
        return -1;
    }

    cout << "beta = alpha^x mod P\n" << beta << " = " << alpha << "^x mod " << P << "\n\n";
    cout << "P = PRODUCT{1, k}(pi^ei)\n" << P << " = ";
    for (i=0; i<k; i++) {
        cout << "(" << primes[i] << "^" << exponents[i] << ")";
        if (i < k-1) {
            cout << " * ";
        } else {
            cout << "\n\n";
        }
    }

//    ZZ x, real_beta;
//    x = pollardLambda(alpha, beta, P, a, b);
//    if (x < 0) {
//        cout << "failure\n";
//    } else {
//        PowerMod(real_beta, alpha, x, P);
//        cout << "x = " << x << endl;
//        cout << real_beta << " = " << alpha << "^" << x << " mod " << P << '\n';
//        assert(real_beta == beta);
//    }
    return 0;
}
