typedef struct {
    int i;
    int type;
    int area;
    double pd;
    double qd;
    double gs;
    double bs;
    double vm;
    double va;
    double baseKV;
    int zone;
    double nvhi;
    double nvlo;
    double evhi;
    double evlo;
} Bus;

typedef struct {
    char id[ID_LEN];   /* generator id in string format         */
    int i;             /* bus id associated with this generator */
    int status;        /* generator status                      */
    double pg;         /* initial real power                    */
    double qg;         /* initial reactive power                */
    double qt;         /* upper bound of reactive power         */
    double qb;         /* lower bound of reactive power         */
    double vg;
    double mBase;
    double pt;         /* upper bound of real power             */
    double pb;         /* lower bound of real power             */
    double pc1;
    double pc2;
    double qc1min;
    double qc1max;
    double qc2min;
    double qc2max;
    double ramp_agc;
    double r;          /* participation factor                  */
    double su_cost;    /* start-up cost                         */
    double sd_cost;    /* shut-down cost                        */
    int obj_type;      /* 1: pwl, 2: coefficient                */
    int nobjcoef;      /* number of objective coefficients      */
    int objcost_start; /* start address of objcost for this gen */
} Generator;

typedef struct {
    char id[ID_LEN];
    int fr;
    int to;
    int status;
    double r;
    double x;
    double b;
    double ratea;
    double rateb;
    double ratec;
    double ratio;
    double angle;
    double angmin;
    double angmax;
    double mag1;
    double mag2;
} Branch;

typedef struct {
    char id[ID_LEN];
    int fr;
    int to;
    int status;
    double r;
    double x;
    double mag1;
    double mag2;
    double ratea;
    double ratec;
    double tap;
    double angle;
} Transformer;

typedef struct {
    int i;
    int status;
    double binit;
    double nb[16];
} Sshunt;

typedef struct {
    char name[64];
    int refcount;
    int input_type;          /* Matpower or GO              */

    int nrefbus;
    int refbus[5];           /* reference bus indices       */
    unsigned nbus;           /* # of buses                  */
    unsigned nload;          /* # of loads                  */
    unsigned ngen;           /* # of generators             */
    unsigned nbranch;        /* # of branches               */
    unsigned ntrans;         /* # of transformers           */
    unsigned nfshunt;        /* # of fixed shunt            */
    unsigned nsshunt;        /* # of switched shunt         */
    unsigned ngencost;       /* # of generators having cost */
    unsigned ngenfactor;     /* # of participation factors  */
    unsigned nctgs;
    unsigned ngenctg;
    unsigned nbractg;

    /* Note active_nbranch also includes active_ntrans.         */
    unsigned active_nload;    /* # of active loads              */
    unsigned active_ngen;     /* # of active generators         */
    unsigned active_nbranch;  /* # of active branches           */
    unsigned active_ntrans;   /* # of active transformers       */
    unsigned active_nfshunt;  /* # of active fixed shunts       */
    unsigned active_nsshunt;  /* # of active switched shunts    */
    unsigned active_ngenpwl;  /* # of active gens with pwl obj  */
    unsigned active_ngenquad; /* # of active gens with quad obj */
    unsigned active_nlimit;   /* # of active line limits        */

    unsigned totpairs;
    unsigned active_totpairs;
    unsigned maxlines_per_bus;

    int *i2b; /* internal bus index i to original bus index b */
    int *b2i; /* original bus index b to internal bus index i */
    int *i2s; /* internal s-shunt index i to original s-shunt index s */
    int *bi2si; /* s-shunt index i of bus i, -1 if no switched shunt */
    int *nfrom; /* nfrom[i]: number of lines leaving from bus i */
    int *nto;   /* nto[i]: number of lines destinating to bus i */
    int *from_to_start;   /* starting address of frombus and tobus */
    int **frombus; /* frombus[i]: line indices leaving from bus i */
    int **tobus;   /* tobus[i]: line indices destinating to bus i */
    int *ngenbus;  /* ngenbus[i]: number of active generators of bus i */
    int *genbus_start;    /* starting address of genbus */
    int **genbus;         /* genbus[i]: active generators connected to bus i */
    int *genpwl_ab_start; /* starting address of pwl_a and pwl_b */
    int *active_genmap;

    double baseMVA;
    Bus *bus;
    Generator *gen;
    Branch *branch;
    Transformer *trans;
    Sshunt *sshunt;

    double *objcost;
    double *pwl_a;
    double *pwl_b;

    double *Y_start;
    double *YttR;
    double *YttI;
    double *YffR;
    double *YffI;
    double *YftR;
    double *YftI;
    double *YtfR;
    double *YtfI;
} Network;

void nw_dealloc(Network **_nw)
{
    Network *nw = (*_nw);

    if (nw->i2b) {
        free(nw->i2b);
    }
    if (nw->b2i) {
        free(nw->b2i);
    }
    if (nw->i2s) {
        free(nw->i2s);
    }
    if (nw->bi2si) {
        free(nw->bi2si);
    }
    if (nw->nfrom) {
        free(nw->nfrom);
    }
    if (nw->nto) {
        free(nw->nto);
    }
    if (nw->from_to_start) {
        free(nw->from_to_start);
    }
    if (nw->frombus) {
        free(nw->frombus);
    }
    if (nw->tobus) {
        free(nw->tobus);
    }
    if (nw->ngenbus) {
        free(nw->ngenbus);
    }
    if (nw->genbus_start) {
        free(nw->genbus_start);
    }
    if (nw->genbus) {
        free(nw->genbus);
    }
    if (nw->active_genmap) {
        free(nw->active_genmap);
    }
    if (nw->bus) {
        free(nw->bus);
    }
    if (nw->gen) {
        free(nw->gen);
    }
    if (nw->branch) {
        free(nw->branch);
    }
    if (nw->trans) {
        free(nw->trans);
    }
    if (nw->sshunt) {
        free(nw->sshunt);
    }
    if (nw->objcost) {
        free(nw->objcost);
        free(nw->pwl_a);
        free(nw->pwl_b);
    }
    if (nw->Y_start) {
        free(nw->Y_start);
    }

    free((*_nw));
    (*_nw) = NULL;
}

Network *nw_get(Network *nw)
{
    nw->refcount++;

    return nw;
}

int nw_put(Network **nw)
{
    int refcount;

    if (nw && (*nw)) {
        (*nw)->refcount--;
        refcount = (*nw)->refcount;

        if ((*nw)->refcount <= 0) {
            nw_dealloc(nw);
        }

        return refcount;
    }

    return Error_NullPointer;
}

Network *nw_alloc()
{
    Network *nw = NULL;

    nw = (Network *)calloc(1, sizeof(*nw));
    nw_get(nw);
    strcpy(nw->name, "default");

    return nw;
}

int parse_mat(Network *nw, const char *filename, const char *delim);

int nw_build_from_matpower(Network *nw, const char *mat)
{
    int rc;

    if ((rc = parse_mat(nw, mat, " \t")) != OK) {
        return rc;
    }
    nw->input_type = Network_Input_Matpower;
    return OK;
}

const char *nw_get_name(const Network *nw)
{
    return nw->name;
}

int nw_set_name(Network *nw, const char *name)
{
    strncpy(nw->name, name, 63);
    nw->name[63] = '\0';
    return OK;
}
