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
#define DIST_BITS 4
#define WILD_KANGAROO false
#define TAME_KANGAROO true
#define WILD_THREADS 2
#define TAME_THREADS 3
#define NUM_THREADS 5

using namespace std;
using namespace NTL;

void inverseMod(ZZ &x, const ZZ &a, const ZZ &m) {
    ZZ tmp1, tmp2, gcd;
    XGCD(tmp1, gcd, tmp2, a, m);
    x = gcd % m;
}

ZZ bruteForceDLP(ZZ alpha, ZZ beta, ZZ P) {
    ZZ res;

    for (res=0; res < P; res++){
        if (PowerMod(alpha, ++res, P) == beta) {
            return res;
        }
    }
    return ZZ(-1);
}

void f(ZZ& x, ZZ& a, ZZ& b, ZZ alpha, ZZ beta, ZZ Q, ZZ P) {
    switch(int(x % 3)) {
        case 1: /* S1 */
            MulMod(x, x, beta, P);
            // a = a;
            AddMod(b, b, 1, Q);
            break;
        case 0: /* S2 */
            MulMod(x, x, x, P);
            MulMod(a, a, 2, Q);
            MulMod(b, b, 2, Q);
            break;
        case 2: /* S3 */
            MulMod(x, x, alpha, P);
            AddMod(a, a, 1, Q);
            // b = b;
            break;
        default:
            exit(1);
    }
}

ZZ pollardRhoSeq(ZZ alpha, ZZ beta, ZZ P, ZZ Q) {
    ZZ res;
    ZZ x, a, b, X, A, B;

    do {
        a = RandomBnd(Q);
    } while (a == 0);
    do {
        b = RandomBnd(Q);
    } while (b == 0);
    x = MulMod(PowerMod(alpha, a, P), PowerMod(beta, b, P), P);

    X = x;
    A = a;
    B = b;

    do {
        f(x, a, b, alpha, beta, Q, P);
        f(X, A, B, alpha, beta, Q, P);
        f(X, A, B, alpha, beta, Q, P);
    } while (x != X);

    if (x == X) {
        ZZ r;
        SubMod(r, b, B, Q);
        if (r == 0) {
            res = ZZ(-1);
        } else {
            ZZ inv_r;
            if (InvModStatus(r, r, Q)) {
                res = ZZ(-1);
//                pollardRho2(alpha, beta, P, Q);
            } else {
                MulMod(res, r, SubMod(A, a, Q), Q);
            }
        }
    }

    return res;
}


ZZ pollardRho(ZZ alpha, ZZ beta, ZZ P, ZZ Q) {
    ZZ x, a, b, a0, b0, X, A, B, res, r;
    bool quit = false;
    map<ZZ, pair<ZZ, ZZ>> distinguished_values;

    #pragma omp parallel \
    num_threads(4) \
    shared(res, quit, distinguished_values) \
    private(x, a, b, r, a0, b0, A, B, X) \
    firstprivate(alpha, beta, P, Q)
    {
        #pragma omp critical
        {
            do {
                a = RandomBnd(Q);
            } while (a == 0);
            do {
                b = RandomBnd(Q);
            } while (b == 0);
            a0 = a;
            b0 = b;
            x = MulMod(PowerMod(alpha, a, P), PowerMod(beta, b, P), P);
        }

        #pragma omp barrier

        do {
            switch (x % 3) {
                case 1: /* S1 */
                    MulMod(x, x, beta, P);
                    // a = a;
                    AddMod(b, b, 1, Q);
                    break;
                case 0: /* S2 */
                    MulMod(x, x, x, P);
                    MulMod(a, a, 2, Q);
                    MulMod(b, b, 2, Q);
                    break;
                case 2: /* S3 */
                    MulMod(x, x, alpha, P);
                    AddMod(a, a, 1, Q);
                    // b = b;
                    break;
                default:
                    exit(1);
            }

            if ((x & ZZ(LeftShift(ZZ(1), DIST_BITS) - ZZ(1))) == 0) {
                #pragma omp critical
                if (!quit) {
                    pair<map<ZZ, pair<ZZ, ZZ>>::iterator, bool> ret = distinguished_values.insert(
                            pair<ZZ, pair<ZZ, ZZ>>(x, pair<ZZ, ZZ>(a0, b0)));
                    if (!ret.second) {
                        quit = true;
                        a = ret.first->second.first;
                        b = ret.first->second.second;
                        x = MulMod(PowerMod(alpha, a, P), PowerMod(beta, b, P), P);
                        X = x;
                        A = a;
                        B = b;

                        do {
                            switch (x % 3) {
                                case 1: /* S1 */
                                    MulMod(x, x, beta, P);
                                    // a = a;
                                    AddMod(b, b, 1, Q);
                                    break;
                                case 0: /* S2 */
                                    MulMod(x, x, x, P);
                                    MulMod(a, a, 2, Q);
                                    MulMod(b, b, 2, Q);
                                    break;
                                case 2: /* S3 */
                                    MulMod(x, x, alpha, P);
                                    AddMod(a, a, 1, Q);
                                    // b = b;
                                    break;
                                default:
                                    exit(1);
                            }
                            switch (X % 3) {
                                case 1: /* S1 */
                                    MulMod(X, X, beta, P);
                                    // A = A;
                                    AddMod(B, B, 1, Q);
                                    break;
                                case 0: /* S2 */
                                    MulMod(X, X, X, P);
                                    MulMod(A, A, 2, Q);
                                    MulMod(B, B, 2, Q);
                                    break;
                                case 2: /* S3 */
                                    MulMod(X, X, alpha, P);
                                    AddMod(A, A, 1, Q);
                                    // B = B;
                                    break;
                                default:
                                    exit(1);
                            }
                            switch (X % 3) {
                                case 1: /* S1 */
                                    MulMod(X, X, beta, P);
                                    // A = A;
                                    AddMod(B, B, 1, Q);
                                    break;
                                case 0: /* S2 */
                                    MulMod(X, X, X, P);
                                    MulMod(A, A, 2, Q);
                                    MulMod(B, B, 2, Q);
                                    break;
                                case 2: /* S3 */
                                    MulMod(X, X, alpha, P);
                                    AddMod(A, A, 1, Q);
                                    // B = B;
                                    break;
                                default:
                                    exit(1);
                            }
                        } while(x != X);

                        SubMod(r, b, B, Q);
                        if (r == 0) {
                            res = ZZ(-1);
                        } else {
                            if (InvModStatus(r, r, Q)) {
                                res = ZZ(-1);
//                                pollardRho(alpha, beta, P, Q);
                            } else {
                                MulMod(res, r, SubMod(A, a, Q), Q);
                            }
                        }
                    }
                }
            }
        } while (!quit);
    }

    distinguished_values.clear();
    return res;
}

ZZ solveSubproblem(const ZZ &alpha, const ZZ &beta, const ZZ &P, const ZZ &Q, const ZZ &q, const ZZ &e) {
    ZZ newAlpha, newBeta, res, gamma, invGamma;

    ZZ tmp1, tmp2, qPowE;
    ZZ l, lastL;
    ZZ j;
    ZZ orderP = (P - ZZ(1)) / Q;

    cout << "\tq=" << q << "; e=" << e << endl;

    res = ZZ(0);
    lastL = ZZ(0);
    qPowE = power(q, conv<long>(e));
    /*
     * newAlpha = alpha^(orderP / q)
     * */
    newAlpha = PowerMod(alpha, orderP / q, P);
    cout << "\tnewAlpha=" << newAlpha << endl;
    gamma = ZZ(1);

    for (j = ZZ(0); j < e; j += ZZ(1)) {
        cout << "\n\tSUBSTEP " << j+1 << "/" << e << endl;
        if (j > 0) {
            /*
             * gamma = gamma * alpha^(q^j-1) mod P
             * */
            gamma = MulMod(gamma, PowerMod(alpha, MulMod(l, PowerMod(q, j - 1, P), P), P), P);
        }
        cout << "\t\tgamma=" << gamma << endl;

        /*
         * newBeta = beta^(gamma^-1) ^ (orderP / (q^j+1))
         * */
        newBeta = PowerMod(MulMod(beta, InvMod(gamma, P), P), orderP / PowerMod(q, j + 1, P), P);
        cout << "\t\tnewBeta=" << newBeta << endl;

        if (newAlpha == newBeta) {
            l = ZZ(1);
        } else if (q == 2) {
            l = bruteForceDLP(newAlpha, newBeta, P);
        } else {
//            l = bruteForceDLP(newAlpha, newBeta, P);
            l = pollardRho(newAlpha, newBeta, P, qPowE);
//            l = pollardRhoSeq(newAlpha, newBeta, P, qPowE);
        }

        if (l < 0) {
            return ZZ(-1);
        }

        cout << "\t\tl_" << j << "=log_" << newAlpha << "(" << newBeta << ") mod " << P << "=" << l << endl;

        PowerMod(tmp1, q, j, P);
        tmp2 = l * tmp1;

        res += tmp2;
        cout << "\t\tx_temp=" << res << endl;
    }

    return res;
}

ZZ chineseReminderTheorem(Vec<ZZ> r, Vec<ZZ> m, int k) {
    ZZ M, tmp, inverse, res;
    ZZ g, x, t;
    int i;

    /* M = m[0] * m[1] * ... * m[len - 1] */
    M = ZZ(1);
    for (i = 0; i < k; ++i) {
        cout << "x=" << r[i] << "\tmod " << m[i] << endl;
        M *= m[i];
    }
    cout << "M=" << M << endl;

    res = ZZ(0);
    for (i = 0; i < k; ++i) {
        tmp = M / m[i];
//        cout << "tmp=" << tmp << "; m_" << i << "=" << m[i] << endl;
        inverseMod(inverse, tmp, m[i]);
        res += (tmp * r[i] * inverse);
        res = res % M;
    }

    return res;
}

ZZ pohligHellman(const ZZ &alpha, const ZZ &beta, const ZZ &P, const ZZ &Q, Vec<ZZ> primes, Vec<ZZ> exponents, int k) {
    Vec<ZZ> xArray;
    Vec<ZZ> pArray;

    ZZ piPowEi;
    ZZ orderP = (P - ZZ(1)) / Q;
    ZZ res;
    int i;

    xArray.SetLength(k);
    pArray.SetLength(k);

    for (i = 0; i < k; ++i) {
        cout << "\nSTEP " << i+1 << "/" << k << endl;
        PowerMod(piPowEi, primes[i], exponents[i], P);

        xArray[i] = solveSubproblem(alpha, beta, P, Q, primes[i], exponents[i]);
        if (xArray[i] < 0) {
            return ZZ(-1);
        }
        cout << "\tx_" << i+1 << "=" << xArray[i] << endl;
        pArray[i] = piPowEi;
    }

    res = chineseReminderTheorem(xArray, pArray, k);
    res = res % orderP;

    return res;
}

void generateInput(ZZ &P, ZZ &Q, Vec<ZZ> &primes, Vec<ZZ> &exponents, long pBits, int n) {
    int i;
    ZZ partP = ZZ(1);
    ZZ q1, q2, q3;

    primes[0] = ZZ(2);
    if (n>1) {
        GenPrime(primes[1], pBits);
        if (n>2) {
            for (i = 2; i < n; i++) {
                NextPrime(primes[i], primes[i-1] + 1);
            }
        }
    }
    for (i = 0; i < n; i++) {
        exponents[i] = RandomBnd(3) + 3;
        partP *= power(primes[i], conv<long>(exponents[i]));
    }

    do {
        GenPrime(q1, 256);
        GenPrime(q2, 256);
        Q = q1 * q2;
        P = (Q * partP) + 1;
    } while (P%2 == 0 || !ProbPrime(P));
}

int main(int argc, char **argv) {

    ZZ Q, P;
    ZZ alpha;
    ZZ beta;
    Vec<ZZ> primes;
    Vec<ZZ> exponents;
    int k, i;

    if (argc == 3) {
        int arg = 1;
        long pBits;

        pBits = conv<long>(argv[arg++]);
        k = conv<int>(argv[arg]);

        primes.SetLength(k);
        exponents.SetLength(k);

        generateInput(P, Q, primes, exponents, pBits, k);

        ZZ r, gen;
        do {
            alpha = RandomBnd(P - ZZ(1)) + ZZ(1);
            PowerMod(alpha, alpha, ZZ(2), P);
        } while (alpha == 1);
        r = RandomBnd(P - ZZ(2)) + ZZ(1);
        PowerMod(beta, alpha, r, P);

        for (i = 0; i < k; ++i) {
            ZZ pe = power(primes[i], conv<long>(exponents[i]));
            cout << "x mod " << pe << " = " << r % pe << endl;
        }

        cout << "Generated data is:" << endl;

        cout << P << endl << endl;
        cout << alpha << endl << endl;
        cout << beta << endl << endl;
        cout << Q << endl << endl;
        for (i=0; i<k; i++) {
            cout << primes[i] << endl << exponents[i] << endl << endl;
        }
    } else if (argc >= 5) {
        int arg = 1;

        P = conv<ZZ>(argv[arg++]);
        alpha = conv<ZZ>(argv[arg++]);
        beta = conv<ZZ>(argv[arg++]);
        Q = conv<ZZ>(argv[arg++]);

        k = argc - 5 >> 1;
        primes.SetLength(k);
        exponents.SetLength(k);

        for (i=0; i<k; ++i) {
            primes[i] = conv<ZZ>(argv[arg++]);
            exponents[i] = conv<ZZ>(argv[arg++]);
        }
    } else {
         cout << "Not enough arguments provided.\n"
                "You have to provide following commandline arguments: hpc_pohlig P Q alpha beta (p e)...\n";
        return -1;
    }

    PowerMod(alpha, alpha, Q, P);
    PowerMod(beta, beta, Q, P);

    cout << "beta^Q = (alpha^Q)^x mod P\n" << beta << " = " << alpha << "^x mod " << P << endl;
    cout << endl << NumBits(Q) << " bits long Q" << endl;
    cout << NumBits(P) << " bits long P" << endl;
    cout << endl << "P - 1 = Q * PRODUCT{1, k}(pi^ei)\n" << P-1 << " = " << Q << " * ";
    ZZ realP;
    realP = Q;
    for (i=0; i<k; i++) {
        cout << "(" << primes[i] << "^" << exponents[i] << ")";
        if (i < k-1) {
            cout << " * ";
        } else {
            cout << endl;
        }
        realP *= PowerMod(primes[i], exponents[i], P);
    }
    assert(P-1 == realP);

    ZZ x, real_beta;
    x = pohligHellman(alpha, beta, P, Q, primes, exponents, k);
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
