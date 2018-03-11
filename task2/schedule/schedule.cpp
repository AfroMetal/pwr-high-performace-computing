#include <omp.h>
#include <cstdio>
#include <cstdlib>
#define Ndim    20
#define Mdim    40
#define Pdim    30

int main (int argc, char *argv[]) {

    int nthreads, tid, i, j, k;
    float *A, *B, *C;
    A = (float *)malloc(Ndim*Pdim*sizeof(float));
    B = (float *)malloc(Pdim*Mdim*sizeof(float));
    C = (float *)malloc(Ndim*Mdim*sizeof(float));
    float tmp;

//    printf("A:\n");
    for (i = 0; i < Ndim; i++) {
        for (k = 0; k < Pdim; k++) {
            tmp = random() % 10;
            *(A + (i*Ndim + k)) = tmp;
//            printf("%f ", tmp);
        }
//        printf("\n");
    }

//    printf("B:\n");
    for (k = 0; k < Pdim; k++) {
        for (j = 0; j < Mdim; j++) {
            tmp = random() % 5;
            *(B + (k*Pdim + j)) = tmp;
//            printf("%f ", tmp);
        }
//        printf("\n");
    }

    #pragma omp parallel for schedule(dynamic) private(tmp, i, j, k)
    for (i = 0; i < Ndim; i++) {
        for (j = 0; j < Mdim; j++) {
                tmp = 0.0;
            for (k = 0; k < Pdim; k++) {
                tmp += *(A + (i*Ndim + k)) * *(B + (k*Pdim + j));
            }
            *(C + (i*Ndim + j)) = tmp;
        }
    }

//    printf("C = A*B:\n");
//    for (i = 0; i < Ndim; i++) {
//        for (j = 0; j < Mdim; j++) {
//            printf("%f ", *(C + (i*Ndim + j)));
//        }
//        printf("\n");
//    }
}