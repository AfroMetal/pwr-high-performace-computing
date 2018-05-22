#include <iostream>
#include <omp.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/vector.h>
#include <cassert>
#include <functional>
#include <string>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <map>

#define DIST_BITS 1

using namespace std;
using namespace NTL;

void inverseMod(ZZ &x, ZZ a, ZZ m) {
    ZZ tmp1, tmp2, gcd;
    XGCD(tmp1, gcd, tmp2, a, m);
    x = gcd % m;
}

ZZ bruteForceDLP(ZZ alpha, ZZ beta, ZZ P) {
    ZZ res = ZZ(0);

    while (PowerMod(alpha, res, P) != beta) {
        res += 1;
    }

    return res;
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

ZZ pollardRho2(ZZ alpha, ZZ beta, ZZ P) {
    ZZ res;
    ZZ Q = (P - ZZ(1)) / ZZ(2);

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
//                res = ZZ(-1);
                pollardRho2(alpha, beta, P);
            } else {
                MulMod(res, r, SubMod(A, a, Q), Q);
            }
        }
    }

    return res;
}

ZZ pollardRho(ZZ alpha, ZZ beta, ZZ P) {
    ZZ x, a, b, a0, b0, X, A, B, res, r;
    bool quit = false;
    ZZ Q = (P - ZZ(1)) / ZZ(2);
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
//                                res = ZZ(-1);
                                pollardRho(alpha, beta, P);
                            } else {
                                MulMod(res, r, SubMod(A, a, Q), Q);
                            }
                        }
                    }
                }
            }
        } while (!quit);
    }

    return res;
}

ZZ solveSubproblem(const ZZ &alpha, const ZZ &beta, const ZZ &P, const ZZ &Q, const ZZ &q, const ZZ &e) {
    ZZ newAlpha, newBeta, res, gamma, invGamma;

    ZZ tmp1, tmp2, tmpMod;
    ZZ l, lastL;
    ZZ j;
    ZZ orderP = (P - ZZ(1)) / Q;

    cout << "q=" << q << "; e=" << e << endl;

    res = ZZ(0);
    lastL = ZZ(0);

    newAlpha = orderP / q;
    PowerMod(newAlpha, alpha, newAlpha, P);
    cout << "newAlpha=" << newAlpha << endl;
    gamma = ZZ(1);

    for (j = ZZ(0); j < e; j += ZZ(1)) {
        cout << "substep " << j+1 << "/" << e << endl;

        PowerMod(tmp1, q, j-1, P);
        tmp1 *= lastL;
        gamma = MulMod(gamma, PowerMod(alpha, tmp1, P), P);
        cout << "gamma=" << gamma << endl;

        InvMod(invGamma, gamma, P);
        PowerMod(tmp1, q, j+1, P);
        tmp1 = orderP / tmp1;
        MulMod(tmp2, beta, invGamma, P);
        PowerMod(newBeta, tmp2, tmp1, P);
        cout << "newBeta=" << newBeta << endl;

        if (newAlpha == newBeta) {
            l = ZZ(1);
//        } else if (NumBits(P) < DIST_BITS) {
//            l = bruteForceDLP(newAlpha, newBeta, P);
        } else {
            l = bruteForceDLP(newAlpha, newBeta, P);
//            l = pollardRho(newAlpha, newBeta, P);
//            l = pollardRho2(newAlpha, newBeta, P);
        }

        if (l < 0) {
            return ZZ(-1);
        }

        cout << "l_" << j << "=log_" << newAlpha << "(" << newBeta << ") mod " << P << "=" << l << endl;

        PowerMod(tmp1, q, j, P);
        tmp2 = l * tmp1;
        lastL = l;

        res += tmp2;
        cout << "x_temp=" << res << endl;
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
//        InvMod(inverse, tmp, m[i]);
        res += (tmp * r[i] * inverse);
        res = res % M;
    }

    return res;
}

ZZ pohligHellman(const ZZ &alpha, const ZZ &beta, const ZZ &P, const ZZ &Q, Vec<ZZ> primes, Vec<ZZ> exponents, int k) {
    Vec<ZZ> xArray;
    Vec<ZZ> pArray;

    ZZ piPowEi;
    ZZ orderP = (P - ZZ(1));
    ZZ res;
    int i;

    xArray.SetLength(k);
    pArray.SetLength(k);

    for (i = 0; i < k; ++i) {
        cout << "step " << i+1 << "/" << k << endl;
        PowerMod(piPowEi, primes[i], exponents[i], P);

        xArray[i] = solveSubproblem(alpha, beta, P, Q, primes[i], exponents[i]);
        if (xArray[i] < 0) {
            return ZZ(-1);
        }
        cout << "x_" << i+1 << "=" << xArray[i] << endl;
        pArray[i] = piPowEi;
    }

    res = chineseReminderTheorem(xArray, pArray, k);
    res = res % orderP;

    return res;
}

void generateInput(ZZ &P, ZZ &Q, Vec<ZZ> &primes, Vec<ZZ> &exponents, long pBits, int n) {
    int i;

    GenPrime(primes[0], pBits);
    for (i = 1; i < n; i++) {
        NextPrime(primes[i], primes[i-1]+1, pBits);
    }

    do {
        GenPrime(Q, pBits*n);
        P = Q;
        for (i = 0; i < n; i++) {
            exponents[i] = RandomBnd(3) + 3;
            P *= power(primes[i], conv<long>(exponents[i]));
        }
        P += 1;
    } while (ProbPrime(P) == 1);
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

    cout << "beta^Q = (alpha^Q)^x mod P\n" << beta << "^" << Q << " = (" << alpha << "^" << Q << ")^x mod " << P << "\n\n";
    cout << NumBits(P) << " bits long P" << endl;
    cout << "P - 1 = Q * PRODUCT{1, k}(pi^ei)\n" << P-1 << " = " << Q << " * ";
    ZZ realP;
    realP = Q;
    for (i=0; i<k; i++) {
        cout << "(" << primes[i] << "^" << exponents[i] << ")";
        if (i < k-1) {
            cout << " * ";
        } else {
            cout << "\n\n";
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
