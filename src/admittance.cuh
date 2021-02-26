int admittance(Network *nw)
{
    int i, line_size;
    double YsR, YsI, YttR, YttI, YffR, YffI, YftR, YftI, YtfR, YtfI;

    const Branch *branch;

    line_size = nw->nbranch;
    nw->Y_start = (double *)calloc((8*line_size), sizeof(double));
    nw->YttR = nw->Y_start;
    nw->YttI = nw->Y_start + line_size;
    nw->YffR = nw->Y_start + 2*line_size;
    nw->YffI = nw->Y_start + 3*line_size;
    nw->YftR = nw->Y_start + 4*line_size;
    nw->YftI = nw->Y_start + 5*line_size;
    nw->YtfR = nw->Y_start + 6*line_size;
    nw->YtfI = nw->Y_start + 7*line_size;

    branch = nw->branch;
    for (i = 0; i < nw->nbranch; i++) {
        double r = branch[i].r;
        double x = branch[i].x;
        double ratio = (branch[i].ratio == 0.0) ? 1.0 : branch[i].ratio;
        double angle = (M_PI/180) * branch[i].angle;

        YsR = branch[i].status * (r / (r*r + x*x));
        YsI = branch[i].status * (-x / (r*r + x*x));
        YttR = YsR;
        YttI = YsI + branch[i].status * (branch[i].b / 2);
        YffR = YttR / (ratio*ratio);
        YffI = YttI / (ratio*ratio);
        YftR = -((YsR*cos(angle) - YsI*sin(angle)) / ratio);
        YftI = -((YsR*sin(angle) + YsI*cos(angle)) / ratio);
        YtfR = -((YsR*cos(angle) + YsI*sin(angle)) / ratio);
        YtfI = -((YsI*cos(angle) - YsR*sin(angle)) / ratio);

        /*
        Ys = branch[i].status / (branch[i].r + _Complex_I*branch[i].x);
        if (branch[i].ratio == 0) {
            tap = cexp(_Complex_I*((M_PI/180) * branch[i].angle));
        } else {
            tap = branch[i].ratio * cexp(1i*((M_PI/180) * branch[i].angle));
        }
        mag = branch[i].status * (branch[i].mag1 + 1i*branch[i].mag2);
        Ytt = Ys + 1i*((branch[i].status * branch[i].b) / 2);
        Yff = (Ytt / (tap * conj(tap))) + mag;
        Yft = -Ys / conj(tap);
        Ytf = -Ys / tap;

        nw->YttR[i] = creal(Ytt);
        nw->YttI[i] = cimag(Ytt);
        nw->YffR[i] = creal(Yff);
        nw->YffI[i] = cimag(Yff);
        nw->YftR[i] = creal(Yft);
        nw->YftI[i] = cimag(Yft);
        nw->YtfR[i] = creal(Ytf);
        nw->YtfI[i] = cimag(Ytf);
        */
        nw->YttR[i] = YttR;
        nw->YttI[i] = YttI;
        nw->YffR[i] = YffR;
        nw->YffI[i] = YffI;
        nw->YftR[i] = YftR;
        nw->YftI[i] = YftI;
        nw->YtfR[i] = YtfR;
        nw->YtfI[i] = YtfI;
    }

    return OK;
}
