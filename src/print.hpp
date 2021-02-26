void print_mat_stat(const Network *nw)
{
    printf("** Statistics of the MATPOWER file\n");
    printf("  System MVA base             : %7.2lf\n", nw->baseMVA);
    printf("  # buses                     : %7d\n", nw->nbus);
    printf("  # loads                     : %7d (%7d active)\n",
           nw->nload, nw->active_nload);
    printf("  # fixed shunts              : %7d (%7d active)\n",
           nw->nfshunt, nw->active_nfshunt);
    printf("  # generators                : %7d (%7d active)\n",
           nw->ngen, nw->active_ngen);
    printf("  # branches                  : %7d (%7d active)\n",
           nw->nbranch, nw->active_nbranch);
    printf("     # transformers           : %7d (%7d active)\n",
           nw->ntrans, nw->active_ntrans);
    printf("  # switched shunts           : %7d (%7d active)\n",
           nw->nsshunt, nw->active_nsshunt);
    printf("\n");
}
