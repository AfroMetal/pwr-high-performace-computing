#include <iostream>
#include <omp.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <assert.h>
#include <tuple>
#include <unordered_map>
#include <map>
#define DIST_BITS 4

using namespace std;
using namespace NTL;


ZZ pollard_rho(ZZ alpha, ZZ beta, ZZ P) {
    int tid;
    ZZ x, a, b, a0, b0, X, A, B, res, r;
    bool quit = false;
    ZZ Q = (P - ZZ(1)) / ZZ(2);
    map<ZZ, pair<ZZ, ZZ>> distinguished_values;

    #pragma omp parallel \
    num_threads(4) \
    shared(res, quit, distinguished_values) \
    private(tid, x, a, b, r, a0, b0, A, B, X) \
    firstprivate(alpha, beta, P, Q)
    {
        tid = omp_get_thread_num();

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

            if ((x & ZZ((1 << DIST_BITS) - 1)) == 0) {
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
                            InvMod(r, r, Q);
                            MulMod(res, r, SubMod(A, a, Q), Q);
                        }
                    }
                }
            }
        } while (!quit);
    }

    return res;
}

int main(int argc, char **argv) {

    ZZ P;
    ZZ alpha;
    ZZ beta;

//    // 40 bits
//    P = 971579802563; // P
//    alpha = 310118604200; // generator g
//    beta = 93968352314; //

    // 45 bits
    P = 30034629688907; // P
    alpha = 15963176245506; // generator g
    beta = 13006361946477; // y = alpha^{x} = beta (mod P)

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
        P = (ZZ(2) * GenGermainPrime_ZZ(atoi(argv[1])-1)) + ZZ(1);

        do {
            alpha = RandomBnd(P - ZZ(1)) + ZZ(1);
            PowerMod(alpha, alpha, ZZ(2), P);
        } while (alpha == 1);

        ZZ r;
        r = RandomBnd(P - ZZ(1)) + ZZ(1);
        PowerMod(beta, alpha, r, P);
    } else {
         cout << "Using default values.\n"
                "You can provide your own as commandline arguments: hpc_pollard [P alpha beta]\n";
    }

    cout << "beta = alpha^x mod P\n" << beta << " = " << alpha << "^x mod " << P << "\n\n";
    ZZ x;
    x = pollard_rho(alpha, beta, P);
    if(x < 0){
        cout << "failure\n";
    } else {
        ZZ real_beta;
        PowerMod(real_beta, alpha, x, P);
        cout << real_beta << " = " << alpha << "^" << x << " mod " << P << '\n';
        assert(real_beta == beta);
    }
    return 0;
}
