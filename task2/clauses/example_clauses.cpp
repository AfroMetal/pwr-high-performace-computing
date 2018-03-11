#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define N       20

int main (int argc, char *argv[]) {

    int nthreads, tid, j, i, n;
    int x;
    float a[N], b[N], c[N];

    /* Some initializations */
    for (i=0; i < N; i++)
        a[i] = b[i] = i;

    for (j=0; j<N;) {
        #pragma omp parallel for firstprivate(j) lastprivate(i)
        for (i=j; i<j+5; i++) {
            c[i] = a[i] + b[i];
            tid = omp_get_thread_num();
            printf("Thread %d: counting c[%d]\n", tid, i);
        }
        j = i;
    }

    if (argc > 1 && atoi(argv[1]) == 1) {
        #pragma omp parallel for private(i,tid) ordered
        for (i=0; i<N; i++) {
            tid = omp_get_thread_num();
            #pragma omp ordered
            {
                printf("Thread %d: c[%d]= %f\n", tid, i, c[i]);
            }
        }
    } else {
        #pragma omp parallel for private(i,tid)
        for (i=0; i<N; i++) {
            tid = omp_get_thread_num();
            printf("Thread %d: c[%d]= %f\n", tid, i, c[i]);
        }
    }

}