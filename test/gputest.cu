#include <iostream>
#include <random>
#include <math.h>

#include "creorder.cuh"
#include "csol.cuh"
#include "cssyax.cuh"
#include "caxpy.cuh"
#include "ccopy.cuh"
#include "cmid.cuh"
#include "cscal.cuh"
#include "cdot.cuh"
#include "ctrqsol.cuh"
#include "cnrm2.cuh"
#include "cgpnorm.cuh"
#include "cgpstep.cuh"
#include "cbreakpt.cuh"
#include "cprsrch.cuh"
#include "ccauchy.cuh"
#include "ctrpcg.cuh"
#include "cicf.cuh"
#include "cicfs.cuh"
#include "cspcg.cuh"
#include "ctron.cuh"
#include "cdriver.cuh"

#include "gputest_utilities.cuh"
#include "gputest_gpnorm.cuh"
#include "gputest_breakpt.cuh"
#include "gputest_trqsol.cuh"
#include "gputest_ssyax.cuh"
#include "gputest_axpy.cuh"
#include "gputest_gpstep.cuh"
#include "gputest_mid.cuh"
#include "gputest_nrm2.cuh"
#include "gputest_scal.cuh"
#include "gputest_copy.cuh"
#include "gputest_dot.cuh"
#include "gputest_trpcg.cuh"
#include "gputest_prsrch.cuh"
#include "gputest_cauchy.cuh"
#include "gputest_icf.cuh"
#include "gputest_icfs.cuh"
#include "gputest_spcg.cuh"
#include "gputest_tron.cuh"
#include "gputest_driver.cuh"

static void usage(const char *progname)
{
    printf("Usage: %s n gridSize\n", progname);
    printf("  n       : the number of variables\n");
    printf("  gridSize: the number of threads in a thread block\n");
}

int main(int argc, char **argv)
{
    int n, gridSize;

    if (argc < 3) {
        usage(argv[0]);
        exit(0);
    }

    n = atoi(argv[1]);
    gridSize = atoi(argv[2]);

    printf(" ** n = %d gridSize = %d \n", n, gridSize);
    test_icf(n, gridSize);
    test_icfs(n, gridSize);
    test_copy(n, gridSize);
    test_dot(n, gridSize);
    test_scal(n, gridSize);
    test_nrm2(n, gridSize);
    test_gpnorm(n, gridSize);
    test_mid(n, gridSize);
    test_gpstep(n, gridSize);
    test_axpy(n, gridSize);
    test_ssyax(n, gridSize);
    test_breakpt(n, gridSize);
    test_cauchy(n, gridSize);
    test_prsrch(n, gridSize);
    test_trpcg(n, gridSize);
    test_trqsol(n, gridSize);
    test_spcg(n, gridSize);
    test_tron(n, gridSize);
    test_driver(n, gridSize);

    return 0;
}