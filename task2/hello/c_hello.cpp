#include <cstdio>
#include <cstdlib>

int main (int argc, char *argv[]) {

    int nthreads = 4, tid;

    for (tid=0; tid<nthreads; tid++) {
        printf("Hello World from thread = %d\n", tid);

/* Only master thread does this */
        if (tid == 0) {
            printf("Number of threads = %d\n", nthreads);
        }

    }
}
