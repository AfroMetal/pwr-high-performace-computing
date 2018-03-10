#include <iostream>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <assert.h>

using namespace std;
using namespace NTL;

void f(ZZ& x, ZZ& a, ZZ& b, ZZ alpha, ZZ beta, ZZ Q, ZZ P) {
    switch(int(x % 3)) {
        case 1: /* S1 */
            x = MulMod(x, beta, P);
            // a = a;
            b = AddMod(b, 1, Q);
            break;
        case 0: /* S2 */
            x = MulMod(x, x, P);
            a = MulMod(a, 2, Q);
            b = MulMod(b, 2, Q);
            break;
        case 2: /* S3 */
            x = MulMod(x, alpha, P);
            a = AddMod(a, 1, Q);
            // b = b;
            break;
        default:
            exit(1);
    }
}

ZZ pollard_rho(ZZ alpha, ZZ beta, ZZ P) {
    ZZ Q = (P - ZZ(1)) / ZZ(2);
    ZZ x, a, b, X, A, B;
    x = 1;
    a = 0;
    b = 0;
    X=x;
    A=a;
    B=b;

    for(ZZ i = ZZ(1); i < Q; ++i ) {
        f(x, a, b, alpha, beta, Q, P);
        f(X, A, B, alpha, beta, Q, P);
        f(X, A, B, alpha, beta, Q, P);

//        cout << i << '\t' << x << '\t' << a << '\t' << b << '\t' << X << '\t' << A << '\t' << B << '\n';

        if(x == X) {
            ZZ r;
            r = SubMod(b, B, Q);
            if(r < 0) r += Q;
            cout << "r = " << r << '\n';
            if(r == 0) {
                return ZZ(-1);
            }
            ZZ inv_r;
            inv_r = InvMod(r, Q);
            if(inv_r < 0) inv_r += Q;
            cout << "inv_r = " << inv_r << '\n';
            ZZ xx;
            xx = MulMod(inv_r, SubMod(A, a, Q), Q);
            if(xx < 0) xx += Q;
            cout << "iters = " << i << "\n\n";
            return xx;
        };
    }
}

int main(int argc, char **argv) {
    ZZ P;
    P = 383; // P
    ZZ alpha;
    alpha = 2; // generator g
    ZZ beta;
    beta = 274; // y = alpha^{x} = beta (mod P)

    // 1907 2 356

    if (argc == 4) {
        P = ZZ(atoi(argv[1]));
        alpha = ZZ(atoi(argv[2]));
        beta = ZZ(atoi(argv[3]));
    } else if (argc == 2) {
        P = (ZZ(2) * GenGermainPrime_ZZ(atoi(argv[1]))) + ZZ(1);

        do {
            alpha = RandomBnd(P - ZZ(1)) + ZZ(1);
            alpha = PowerMod(alpha, ZZ(2), P);
        } while (alpha == ZZ(1));

        ZZ r;
        r = RandomBnd(P - ZZ(1)) + ZZ(1);
        beta = PowerMod(alpha, r, P);
    } else {
        cout << "Using default values.\n"
                "You can provide your own as commandline arguments: hpc_pollard [P alpha beta]\n";
    }

    cout << "Given prime P = (q - 1) / 2, Z_{P} group generator alpha, and beta "
                   "such that beta = alpha^x mod P is sufficed for some x, find that x.\n\n";

    cout << "beta = alpha^x mod P\n" << beta << " = " << alpha << "^x mod " << P << "\n\n";

    ZZ x = pollard_rho(alpha, beta, P);
    if(x < 0){
        std::cout << "failure\n";
    } else {
        ZZ real_beta = PowerMod(alpha, x, P);
        cout << "x = " << x << '\n';
        cout << real_beta << " = " << alpha << "^" << x << " mod " << P << '\n';
        assert(real_beta == beta);
    }
    return 0;
}
