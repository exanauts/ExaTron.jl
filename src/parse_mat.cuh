#define ParseMat_Start       0x1f
#define ParseMat_MVA         0x1
#define ParseMat_Bus         0x2
#define ParseMat_Generator   0x4
#define ParseMat_Branch      0x8
#define ParseMat_Gencost     0x10

#if 0
typedef enum
{
    ParseMat_Start = 0,
    ParseMat_MVA,
    ParseMat_Bus,
    ParseMat_Generator,
    ParseMat_Branch,
    ParseMat_Gencost
} ParseMatStage;
#endif

static int skipsection(FILE *fp, char **line, size_t *linecap)
{
    int pos, num_skipped = 0;
    char *buf;

    while (getline(line, linecap, fp) > 0) {
        buf = (*line);
        pos = 0;
        while (isspace(buf[pos])) pos++;

        if (buf[pos] == ']' && buf[pos+1] == ';') {
            break;
        }

        if (buf[pos] != '\0') {
            num_skipped++;
        }
    }

    return num_skipped;
}
static int read_bus(Network *nw, FILE *fp, char **line, size_t *linecap,
                    const char *delim)
{
    long fpos;
    int i;
    ssize_t n_lineread;

    fpos = ftell(fp);
    nw->nbus = skipsection(fp, line, linecap);
    nw->bus = (Bus *)calloc(MAX(nw->nbus, 1), sizeof(Bus));
    if (nw->nbus <= 0) {
        return Error_InvalidFileFormat;
    }
    fseek(fp, fpos, SEEK_SET);

    nw->nrefbus = 0;
    nw->active_nload = 0;
    nw->active_nfshunt = 0;
    for (i = 0; i < nw->nbus; i++) {
        n_lineread = getline(line, linecap, fp);
        assert(n_lineread > 0);

        nw->bus[i].i = atoi(strtok(*line, delim));
        nw->bus[i].type = atoi(strtok(NULL, delim));

        if (nw->bus[i].type == 3) {
            if (nw->nrefbus == 5) {
                printf("Error: more than 5 reference buses are not allowed.\n");
                assert(False);
            }
            nw->refbus[nw->nrefbus] = i;
            nw->nrefbus++;
        }

        nw->bus[i].pd = atof(strtok(NULL, delim));
        nw->bus[i].qd = atof(strtok(NULL, delim));
        if (nw->bus[i].pd != 0 || nw->bus[i].qd != 0) {
            nw->active_nload++;
        }

        nw->bus[i].gs = atof(strtok(NULL, delim));
        nw->bus[i].bs = atof(strtok(NULL, delim));
        if (nw->bus[i].gs != 0 || nw->bus[i].bs != 0) {
            nw->active_nfshunt++;
        }

        nw->bus[i].area = atoi(strtok(NULL, delim));
        nw->bus[i].vm = atof(strtok(NULL, delim));
        nw->bus[i].va = atof(strtok(NULL, delim));
        nw->bus[i].baseKV = atof(strtok(NULL, delim));
        nw->bus[i].zone = atoi(strtok(NULL, delim));
        nw->bus[i].nvhi = atof(strtok(NULL, delim));
        nw->bus[i].nvlo = atof(strtok(NULL, delim));
        nw->bus[i].evhi = 0;
        nw->bus[i].evlo = 0;
    }

    if (nw->nrefbus == 0) {
        printf("Error: no refereuce bus was found.\n");
        assert(False);
    }

    if (nw->nrefbus > 1) {
        printf("Warning: more than one reference bus was found.\n");
    }

    nw->nload = nw->active_nload;
    nw->nfshunt = nw->active_nfshunt;

    n_lineread = getline(line, linecap, fp); /* the last line "];" */
    assert(n_lineread > 0);
    return OK;
}

static int read_generator(Network *nw, FILE *fp, char **line,
                          size_t *linecap, const char *delim)
{
    long fpos;
    int i, active;
    ssize_t n_lineread;
    const char *tok;

    assert(nw->nbus > 0 && nw->b2i != NULL);

    fpos = ftell(fp);
    nw->ngen = skipsection(fp, line, linecap);
    nw->gen = (Generator *)calloc(MAX(nw->ngen, 1), sizeof(Generator));
    nw->ngenbus = (int *)calloc(nw->nbus, sizeof(int));
    nw->active_genmap = (int *)calloc(MAX(nw->ngen, 1), sizeof(int));
    if (nw->ngen <= 0) {
        return Error_InvalidFileFormat;
    }
    fseek(fp, fpos, SEEK_SET);

    active = 0;
    for (i = 0; i < nw->ngen; i++){
        n_lineread = getline(line, linecap, fp);
        assert(n_lineread > 0);

        nw->gen[active].i = atoi(strtok(*line, delim));
        nw->gen[active].pg = atof(strtok(NULL, delim));
        nw->gen[active].qg = atof(strtok(NULL, delim));

        tok = strtok(NULL, delim);
        nw->gen[active].qt = !strcasecmp(tok, "inf") ? 99999.0 : atof(tok);
        /*nw->gen[i].qt = !strcasecmp(tok, "inf") ? INF : atof(tok);*/
        tok = strtok(NULL, delim);
        nw->gen[active].qb = !strcasecmp(tok, "-inf") ? -99999.0 : atof(tok);
        /*nw->gen[i].qb = !strcasecmp(tok, "-inf") ? -INF : atof(tok);*/

        nw->gen[active].vg = atof(strtok(NULL, delim));
        nw->gen[active].mBase = atof(strtok(NULL, delim));
        nw->gen[active].status = atoi(strtok(NULL, delim));

        tok = strtok(NULL, delim);
        nw->gen[active].pt = !strcasecmp(tok, "inf") ? 99999.0 : atof(tok);
        /*nw->gen[i].pt = !strcasecmp(tok, "inf") ? INF : atof(tok);*/
        tok = strtok(NULL, delim);
        nw->gen[active].pb = !strcasecmp(tok, "-inf") ? -99999.0 : atof(tok);
        /*nw->gen[i].pb = !strcasecmp(tok, "-inf") ? -INF : atof(tok);*/

        if (strchr(tok, ';') == NULL) {
            nw->gen[active].pc1 = atof(strtok(NULL, delim));
            nw->gen[active].pc2 = atof(strtok(NULL, delim));
            nw->gen[active].qc1min = atof(strtok(NULL, delim));
            nw->gen[active].qc1max = atof(strtok(NULL, delim));
            nw->gen[active].qc2min = atof(strtok(NULL, delim));
            nw->gen[active].qc2max = atof(strtok(NULL, delim));
            nw->gen[active].ramp_agc = atof(strtok(NULL, delim));
        }

        if (nw->gen[active].status == 1){
            nw->ngenbus[nw->b2i[nw->gen[active].i]]++;
            nw->active_genmap[active] = i;
            active++;
        }
    }
    nw->active_ngen = active;

    n_lineread = getline(line, linecap, fp); /* the last line "];" */
    assert(n_lineread > 0);
    return OK;
}

static int read_branch(Network *nw, FILE *fp, char **line,
                       size_t *linecap, const char *delim)
{
    long fpos;
    int i, active, fr, to;
    ssize_t n_lineread;

    fpos = ftell(fp);
    nw->nbranch = skipsection(fp, line, linecap);
    nw->branch = (Branch *)calloc(MAX(nw->nbranch, 1), sizeof(Branch));
    nw->nfrom = (int *)calloc(nw->nbus, sizeof(int));
    nw->nto = (int *)calloc(nw->nbus, sizeof(int));
    if (nw->nbranch <= 0) {
        return Error_InvalidFileFormat;
    }
    fseek(fp, fpos, SEEK_SET);

    active = 0;
    for (i = 0; i < nw->nbranch; i++) {
        n_lineread = getline(line, linecap, fp);
        assert(n_lineread > 0);

        fr = atoi(strtok(*line, delim));
        to = atoi(strtok(NULL, delim));
        nw->branch[active].fr = fr;
        nw->branch[active].to = to;
        nw->branch[active].r = atof(strtok(NULL, delim));
        nw->branch[active].x = atof(strtok(NULL, delim));
        nw->branch[active].b = atof(strtok(NULL, delim));
        nw->branch[active].ratea = atof(strtok(NULL, delim));
        nw->branch[active].rateb = atof(strtok(NULL, delim));
        nw->branch[active].ratec = atof(strtok(NULL, delim));
        nw->branch[active].ratio = atof(strtok(NULL, delim));
        nw->branch[active].angle = atof(strtok(NULL, delim));
        nw->branch[active].status = atoi(strtok(NULL, delim));
        nw->branch[active].angmin = atof(strtok(NULL, delim));
        nw->branch[active].angmax = atof(strtok(NULL, delim));

        if (nw->branch[active].ratio != 0 || nw->branch[active].angle != 0) {
            nw->ntrans++;
        }

        if (nw->branch[active].status == 1) {
            nw->nfrom[nw->b2i[fr]]++;
            nw->nto[nw->b2i[to]]++;

            if (nw->branch[active].ratio != 0 || nw->branch[active].angle != 0) {
                nw->active_ntrans++;
            }
            if (nw->branch[active].ratea != 0 && nw->branch[active].ratea < 1e10) {
                nw->active_nlimit++;
            }
            active++;
        }
    }
    nw->active_nbranch = active;

    n_lineread = getline(line, linecap, fp);
    assert(n_lineread > 0);
    return OK;
}

static int read_gencost(Network *nw, FILE *fp, char **line,
                        size_t *linecap, const char *delim)
{
    int i, j, loc, obj_type, nobjcoef, nread, active;
    ssize_t n_lineread;

    loc = 0;
    nw->active_ngenpwl = 0;
    nw->active_ngenquad = 0;
    nw->objcost = NULL;
    active = 0;
    for (i = 0; i < nw->ngen; i++) {
        n_lineread = getline(line, linecap, fp);
        assert(n_lineread > 0);

        if (nw->active_genmap[active] != i) {
            continue;
        }

        nw->gen[active].obj_type = atoi(strtok(*line, delim));
        nw->gen[active].su_cost = atof(strtok(NULL, delim));
        nw->gen[active].sd_cost = atof(strtok(NULL, delim));
        nw->gen[active].nobjcoef = atoi(strtok(NULL, delim));

        if (!nw->objcost) {
            if (nw->gen[active].obj_type == Objective_Quadratic) {
                obj_type = nw->gen[active].obj_type;
                nobjcoef = nw->gen[active].nobjcoef;
                nw->objcost = (double *)calloc(nobjcoef*MAX(nw->active_ngen, 1), sizeof(double));
            } else {
                obj_type = nw->gen[active].obj_type;
                nobjcoef = nw->gen[active].nobjcoef;
                nw->objcost = (double *)calloc(2*nobjcoef*MAX(nw->active_ngen, 1), sizeof(double));
            }
        }

        assert(nobjcoef == nw->gen[active].nobjcoef);
        assert(obj_type == nw->gen[active].obj_type);

        nw->gen[active].objcost_start = loc;
        nread = nobjcoef;
        if (obj_type == Objective_PiecewiseLinear) {
            nread *= 2;
        }
        for (j = 0; j < nread; j++) {
            nw->objcost[loc++] = atof(strtok(NULL, delim));
        }

        if (obj_type == Objective_Quadratic) {
            nw->active_ngenquad++;
        } else {
            nw->active_ngenpwl++;
        }

        active++;
    }

    return OK;
}

static int parse_build_busmap(Network *nw)
{
    int i, busid, max_busid;

    max_busid = 0;
    for (i = 0; i < nw->nbus; i++) {
        busid = nw->bus[i].i;
        if (max_busid < busid) {
            max_busid = busid;
        }
    }

    nw->i2b = (int *)calloc(nw->nbus, sizeof(int));
    nw->b2i = (int *)calloc(max_busid + 1, sizeof(int));
    for (i = 0; i < nw->nbus; i++) {
        nw->i2b[i] = nw->bus[i].i;
        nw->b2i[nw->i2b[i]] = i;
    }

    return OK;
}

static int parse_build_genbusmap(Network *nw)
{
    int i, loc, bus;

    nw->genbus_start = (int *)calloc(MAX(1, nw->active_ngen), sizeof(int));
    nw->genbus = (int **)calloc(nw->nbus, sizeof(int *));

    loc = 0;
    for (i = 0; i < nw->nbus; i++) {
        nw->genbus[i] = nw->genbus_start + loc;
        loc += nw->ngenbus[i];
        nw->ngenbus[i] = 0;
    }

    for (i = 0; i < nw->active_ngen; i++) {
        bus = nw->b2i[nw->gen[i].i];
        *(nw->genbus[bus] + nw->ngenbus[bus]) = i;
        nw->ngenbus[bus]++;
    }

    return OK;
}

static int parse_build_fromtomap(Network *nw)
{
    int i, loc, size, fr, to, maxlines_per_bus;

    size = 2*nw->active_nbranch;
    nw->from_to_start = (int *)calloc(MAX(1, size), sizeof(int));
    nw->frombus = (int **)calloc(nw->nbus, sizeof(int *));
    nw->tobus = (int **)calloc(nw->nbus, sizeof(int *));

    /* Initialize the starting address of each bus' from and to info. */
    loc = 0;
    for (i = 0; i < nw->nbus; i++) {
        nw->frombus[i] = nw->from_to_start + loc;
        loc += nw->nfrom[i];
        nw->tobus[i] = nw->from_to_start + loc;
        loc += nw->nto[i];
        nw->nfrom[i] = 0;
        nw->nto[i] = 0;
    }

    for (i = 0; i < nw->active_nbranch; i++) {
        fr = nw->b2i[nw->branch[i].fr];
        to = nw->b2i[nw->branch[i].to];
        *(nw->frombus[fr] + nw->nfrom[fr]) = i;
        *(nw->tobus[to] + nw->nto[to]) = i;
        nw->nfrom[fr]++;
        nw->nto[to]++;
    }

    /* Identify the maximum number of lines per bus for statistics reason. */
    maxlines_per_bus = 0;
    for (i = 0; i < nw->nbus; i++) {
        if (maxlines_per_bus < nw->nfrom[i] + nw->nto[i]) {
            maxlines_per_bus = nw->nfrom[i] + nw->nto[i];
        }
    }
    nw->maxlines_per_bus = maxlines_per_bus;

    return OK;
}

int parse_mat(Network *nw, const char *filename, const char *delim)
{
    int pos, rc = OK;
    FILE *fp = fopen(filename, "r");
    char *line = NULL;
    char keyword[64];
    size_t linecap = 1024;
    int stage;

    line = (char *)calloc(linecap, sizeof(char));

    if (!fp) {
        printf("File not found: %s.\n", filename);
        return Error_NotFound;
    }

    stage = ParseMat_Start;
    while(getline(&line, &linecap, fp) > 0) {
        if (line[0] != '%') {
            pos = 0;
            while (!isspace(line[pos])) pos++;
            strncpy(keyword, line, pos);
            keyword[pos] = '\0';

            if (!strcmp(keyword, "mpc.baseMVA")) {
                strtok(line, delim);
                strtok(NULL, delim);
                nw->baseMVA = atof(strtok(NULL, delim));
                stage &= (~ParseMat_MVA);
            } else if (!strcmp(keyword, "mpc.bus")) {
                rc = read_bus(nw, fp, &line, &linecap, delim);
                if (rc == OK) {
                    parse_build_busmap(nw);
                    stage &= (~ParseMat_Bus);
                } else {
                    goto out;
                }
            } else if (!strcmp(keyword, "mpc.gen")) {
                rc = read_generator(nw, fp, &line, &linecap, delim);
                if (rc == OK) {
                    parse_build_genbusmap(nw);
                    stage &= (~ParseMat_Generator);
                } else {
                    goto out;
                }
            } else if (!strcmp(keyword, "mpc.branch")) {
                rc = read_branch(nw, fp, &line, &linecap, delim);
                if (rc == OK) {
                    parse_build_fromtomap(nw);
                    stage &= (~ParseMat_Branch);
                } else {
                    goto out;
                }
            } else if (!strcmp(keyword, "mpc.gencost")) {
                rc = read_gencost(nw, fp, &line, &linecap, delim);
                if (rc == OK) {
                    stage &= (~ParseMat_Gencost);
                } else {
                    goto out;
                }
            }
        }
    }

out:
    if (stage != 0) {
        rc = Error_InvalidFileFormat;
    }

    free(line);
    fclose(fp);

    return rc;
}
