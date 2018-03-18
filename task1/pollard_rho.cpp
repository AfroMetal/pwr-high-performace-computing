#include <iostream>
#include <omp.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <assert.h>
#include <tuple>

using namespace std;
using namespace NTL;

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

ZZ pollard_rho(ZZ alpha, ZZ beta, ZZ P) {
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
            InvMod(inv_r, r, Q);
            MulMod(res, inv_r, SubMod(A, a, Q), Q);\
        }
    }

    return res;
}

int main(int argc, char **argv) {

    ZZ P;
    ZZ alpha;
    ZZ beta;

//    // 40 bits
//    P = 133784127887; // P
//    alpha = 125752947375; // generator g
//    beta = 15994972089; // y = alpha^{x} = beta (mod P)

    // 50 bits
    P = 1869034281506423; // P
    alpha = 834224574754024; // generator g
    beta = 889899983492440; // y = alpha^{x} = beta (mod P)

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
