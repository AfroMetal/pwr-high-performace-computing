#include <cstdio>
#include <cstdlib>
#define N       1000

int main (int argc, char *argv[]) {

    int nthreads = 4, tid, i;
    float a[N], b[N], c[N];

    /* Some initializations */
    for (i=0; i < N; i++)
        a[i] = b[i] = i;

    for (tid=0; tid<nthreads; tid++) {
        if (tid == 0) {
            printf("Number of threads = %d\n", nthreads);
        }

        printf("Thread %d starting...\n",tid);

        for (i=0; i<N; i++) {
            c[i] = a[i] + b[i];
            printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
        }

    }
}