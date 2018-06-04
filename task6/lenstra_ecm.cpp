#include <iostream>
#include <omp.h>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/vector.h>
#include <assert.h>
#include <functional>
#include <string>

#define NUM_THREADS 4

using namespace std;
using namespace NTL;

struct Point
{
    ZZ x;
    ZZ y;
    ZZ z;
    Point() {
        this->x = ZZ(0);
        this->y = ZZ(1);
        this->z = ZZ(0);
    };
};

struct EllipticCurve
{
    ZZ a;
    ZZ b;
    ZZ n;
    EllipticCurve(const Point* p, const ZZ &N) {
        this->a = RandomBnd(N);
        this->n = N;

        ZZ y2, x3, ax;
        // b = y_p^2 - x^3 - a*x mod N
        PowerMod(y2, p->y, 2, N);
        PowerMod(x3, p->x, 3, N);
        MulMod(ax, this->a, p->x, N);
        SubMod(this->b, SubMod(y2, x3, N), ax, N);
    }
};


Point* randomPoint(const ZZ &N) {
    auto* p = new Point;
    p->x = RandomBnd(N);
    p->y = RandomBnd(N);
    p->z = 1;
    return p;
}

Point* addPoint(const Point* p, const Point* q, const EllipticCurve* ecurve) {
    Point* r = new Point;
    if (p->z == 0) {
        // r = O + q = q
        r->x = q->x;
        r->y = q->y;
        r->z = q->z;
        return r;
    }
    if (q->z == 0) {
        // r = p + O = p
        r->x = p->x;
        r->y = p->y;
        r->z = p->z;
        return r;
    }
    ZZ top, bottom;
    if (p->x == q->x) {
        if (AddMod(p->y, q->y, ecurve->n) == 0) {
            // points are opposite to each other, return O
            return r;
        }
        // lambda = (3 * x_p^2 + a) / (2 * y_p) mod N
        AddMod(top, 3 * (p->x * p->x), ecurve->a, ecurve->n);
        MulMod(bottom, p->y, 2, ecurve->n);
    } else {
        // lambda = (y_q - y_p) / (x_q - x_p) mod N
        SubMod(top, q->y, p->y, ecurve->n);
        SubMod(bottom, q->x, p->x, ecurve->n);
    }
    // x_r = lambda^2 - x_p - x_q mod N
    ZZ inv;
    if (InvModStatus(inv, bottom, ecurve->n)) {
        r->x = 0;
        r->y = 0;
        r->z = bottom;
        return r;
    }

    ZZ lambda;
    MulMod(lambda, top, inv, ecurve->n);

    // x_r = lambda^2 - x_p - x_q
    MulMod(r->x, lambda, lambda, ecurve->n);
    SubMod(r->x, r->x, p->x, ecurve->n);
    SubMod(r->x, r->x, q->x, ecurve->n);

    // y_r = lambda * (x_p - x_r) - y_p mod N
    SubMod(r->y, p->x, r->x, ecurve->n);
    MulMod(r->y, lambda, r->y, ecurve->n);
    SubMod(r->y, r->y, p->y, ecurve->n);

    r->z = 1;
    return r;
}

void addPoint(Point* r, const Point* p, const Point* q, const EllipticCurve* ecurve) {
    Point* res = addPoint(p, q, ecurve);
    r->x = res->x;
    r->y = res->y;
    r->z = res->z;
    delete res;
}

void mulPoint(Point* r, const Point* p, unsigned k, const EllipticCurve* ecurve) {
    Point* q = new Point;
    q->x = p->x;
    q->y = p->y;
    q->z = p->z;
    r->x = 0;
    r->y = 1;
    r->z = 0;
    while (k > 0) {
        if (p->z > 1) {
            r->x = p->x;
            r->y = p->y;
            r->z = p->z;
            return;
        }
        if (k & 1) {
            addPoint(r, q, r, ecurve);
        }
        k >>= 1;
        addPoint(q, q, q, ecurve);
    }
    delete q;
}

Point* mulPoint(const Point* p, unsigned k, const EllipticCurve* ecurve) {
    Point* r = new Point;
    mulPoint(r, p, k, ecurve);
    return r;
}

//Vec<ZZ> sieve(ZZ n) {
//
//}


ZZ lenstraECM(unsigned B, const ZZ &N) {
    int tid;
    Point* point;
    EllipticCurve* ecurve;
    ZZ x = ZZ(-1);
    bool quit = false;

    #pragma omp parallel \
    num_threads(NUM_THREADS) \
    shared(x, quit, N) \
    private(tid, point, ecurve) \
    firstprivate(B)
    {
        tid = omp_get_thread_num();
        switch (tid) {
            case 0:
                B /= 100;
                break;
            case NUM_THREADS:
                B *= 100;
                break;
            default:
                break;
        }
        ZZ g, check;
        do {
            do {
                point = randomPoint(N);
                ecurve = new EllipticCurve(point, N);
                MulMod(check, 4, PowerMod(ecurve->a, 3, ecurve->n), ecurve->n);
                AddMod(check, check, MulMod(27, MulMod(ecurve->b, ecurve->b, ecurve->n), ecurve->n), ecurve->n);
                g = GCD(check, ecurve->n);
            } while (g == N);

            if (g > 1) {
                #pragma critical
                if (!quit) {
                    quit = true;
                    cout << "thread " << tid << " found the result!" << endl;
                    x = g;
                };
            } else {
                unsigned k;
                for (k = 2; k <= B; k++) {
                    if (!quit) {
                        mulPoint(point, point, k, ecurve);
                        if (point->z > 1) {
                            #pragma critical
                            if (!quit) {
                                quit = true;
                                cout << "thread " << tid << " found the result!" << endl;
                                x = GCD(point->z, ecurve->n);
                            }
                        }
                    } else {
                        k = B + 1;
                    }
                }
            }
        } while (!quit);
    }

    return x;
}


int main(int argc, char **argv) {

    ZZ N;
    long n;
    unsigned B;
    Vec<ZZ> primes;

    if (argc == 3) {
        long arg = 1;
        n = conv<long>(argv[arg++]);
        N = GenPrime_ZZ(n) * GenPrime_ZZ(n);
        B = conv<unsigned>(argv[arg]);

        cout << "N = " << N << endl;
        cout << "B = " << B << endl;
    } else {
         cout << "Not enough arguments provided." << endl <<
              "You have to provide following commandline arguments: hpc_lenstra N B" << endl;
        return -1;
    }

    ZZ x = lenstraECM(B, N);

    if (x < 0) {
        cout << "failure" << endl;
        return -1;
    } else {
        assert(N % x == 0);
    }

    cout << N << " = " << x << " * " << N/x << endl;

    return 0;
}