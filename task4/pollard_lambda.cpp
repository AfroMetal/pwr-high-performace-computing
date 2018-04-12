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

ZZ pollard_lambda(ZZ alpha, ZZ beta, ZZ P, ZZ a, ZZ b) {
    int tid;
    Vec<ZZ> dists;
    Vec<ZZ> jumps;
    unsigned long r;
    int index;
    string str;
    ZZ Q = (P - ZZ(1)) / ZZ(2);
    ZZ res, beta_min, vi;
    ZZ order;
    order = P - ZZ(1);
    bool quit = false;
    map<ZZ, tuple<ZZ, bool, int>> distinguished_values;

    // beta_min = NUM_THREADS * sqrt(b - a) / 4
    beta_min = MulMod(ZZ(NUM_THREADS), SqrRoot(b - a), P) / ZZ(4);
    // vi = beta_min / NUM_THREADS / 2
    vi = beta_min / ZZ(NUM_THREADS) / ZZ(2);

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

    bool kangaroo_type;
    ZZ dist, pos, x, step;

    #pragma omp parallel \
    num_threads(NUM_THREADS) \
    shared(res, quit, distinguished_values, jumps, dists, r, vi) \
    private(tid, dist, pos, kangaroo_type, x, step, index, str) \
    firstprivate(a, b, alpha, beta, P, Q)
    {
        tid = omp_get_thread_num();

        #pragma omp critical
        {
            dist = 0;

            if (tid < TAME_THREADS) {
                kangaroo_type = TAME_KANGAROO;
                ZZ i;
                i = tid;
                // dist = i*vi
//                dist = MulMod(i, vi, Q);
                // pos = g^((a + b) / 2 + iv)
                pos = PowerMod(alpha, (((a + b) / ZZ(2)) + i*ZZ(WILD_THREADS)) % P, P);
            } else {
                kangaroo_type = WILD_KANGAROO;
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
                    pair<map<ZZ, tuple<ZZ, bool, int>>::iterator, bool> ret = distinguished_values.insert(
                            pair<ZZ, tuple<ZZ, bool, int>>(pos, make_tuple(dist, kangaroo_type, tid)));
                    if (!ret.second) {
                        quit = true;
//                        cout << "collision:" << get<2>(ret.first->second) << ": " << ret.first->first << endl;
                        // x = (a + b) / 2 + iv - ju + dist_TAME - dist_WILD
                        x = ZZ(a + b) / ZZ(2);
                        ZZ i, j, dist_tame, dist_wild;
                        if (kangaroo_type == TAME_KANGAROO) {
                            i = tid;
                            j = get<2>(ret.first->second) - TAME_THREADS;
                            dist_tame = dist % P;
                            dist_wild = get<0>(ret.first->second) % P;
                        } else {
                            i = get<2>(ret.first->second);
                            j = tid - TAME_THREADS;
                            dist_tame = get<0>(ret.first->second) % P;
                            dist_wild = dist % P;
                        }
                        AddMod(x, x, dist_tame, P);
                        SubMod(x, x, dist_wild, P);
                        AddMod(x, x, i * WILD_THREADS, P);
                        SubMod(x, x, j * TAME_THREADS, P);

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

    // 40 bits
    P = 971579802563; // P
    alpha = 310118604200; // generator g
    beta = 93968352314; //

//    // 45 bits
//    P = 30034629688907; // P
//    alpha = 15963176245506; // generator g
//    beta = 13006361946477; // y = alpha^{x} = beta (mod P)

//    // 50 bits
//    P = 1869034281506423; // P
//    alpha = 834224574754024; // generator g
//    beta = 889899983492440; // y = alpha^{x} = beta (mod P)

    if (argc == 4) {
        P = ZZ(atoi(argv[1]));
        alpha = ZZ(atoi(argv[2]));
        beta = ZZ(atoi(argv[3]));
    } else if (argc == 2) {
        cout << "Using random " << argv[1] << " bits P and random values.\n";
        Q = GenGermainPrime_ZZ(atoi(argv[1])-1);
        P = (ZZ(2) * Q) + ZZ(1);

        do {
            alpha = RandomBnd(P - ZZ(1)) + ZZ(1);
            PowerMod(alpha, alpha, ZZ(2), P);
        } while (alpha == 1);

        // range [0, (pi / 8) * ord g]
        // ord g = Q for P = (2 * Q) + 1
        a = 0;
        b = (ZZ(314 / 8) * Q) / 100;

        ZZ r;
        r = RandomBnd(b - ZZ(1)) + ZZ(1);
        r += a;
        cout << "[a, b] = [" << a << ", " << b << "]" << endl;
//        cout << "x = " << r << endl;
        PowerMod(beta, alpha, r, P);
    } else {
         cout << "Using default values.\n"
                "You can provide your own as commandline arguments: hpc_pollard [P alpha beta]\n";
    }

    cout << "beta = alpha^x mod P\n" << beta << " = " << alpha << "^x mod " << P << "\n\n";
    ZZ x, real_beta;
    x = pollard_lambda(alpha, beta, P, a, b);
    if (x < 0) {
        cout << "failure\n";
    } else {
        PowerMod(real_beta, alpha, x, P);
        cout << "x = " << x << endl;
        cout << real_beta << " = " << alpha << "^" << x << " mod " << P << '\n';
        assert(real_beta == beta);
    }
    return 0;
}
