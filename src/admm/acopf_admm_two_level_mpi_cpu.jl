function admm_restart_two_level_mpi(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}};
    outer_iterlim=outer_iterlim, inner_iterlim=inner_iterlim, scale=scale
)
    par = env.params
    sol = mod.solution

    # -------------------------------------------------------------------
    # Variables are of two types: u and v
    #   u contains variables for generators and branches and
    #   v contains variables for buses.
    #
    # Variable layout:
    #   u: | (pg,qg)_g | (pij,qij,pji,qji,w_i_ij,w_j_ij,a_i_ij, a_i_ji)_ij |
    #   v: | (pg,qg)_g | (pij,qij,pji,qji)_ij | (w_i,theta_i)_i |
    #   xbar: same as v
    # -------------------------------------------------------------------

    x_curr = sol.x_curr
    xbar_curr = sol.xbar_curr
    z_outer = sol.z_outer
    z_curr = sol.z_curr
    z_prev = sol.z_prev
    l_curr = sol.l_curr
    lz = sol.lz
    rho = sol.rho
    rp = sol.rp
    rd = sol.rd
    rp_old = sol.rp_old
    Ax_plus_By = sol.Ax_plus_By
    wRIij = sol.wRIij

    u_curr = view(x_curr, 1:mod.nvar_u)
    v_curr = view(x_curr, mod.nvar_u_padded+1:mod.nvar)
    zu_curr = view(z_curr, 1:mod.nvar_u)
    zv_curr = view(z_curr, mod.nvar_u_padded+1:mod.nvar)
    lu_curr = view(l_curr, 1:mod.nvar_u)
    lv_curr = view(l_curr, mod.nvar_u_padded+1:mod.nvar)
    lz_u = view(lz, 1:mod.nvar_u)
    lz_v = view(lz, mod.nvar_u_padded+1:mod.nvar)
    rho_u = view(rho, 1:mod.nvar_u)
    rho_v = view(rho, mod.nvar_u_padded+1:mod.nvar)
    rp_u = view(rp, 1:mod.nvar_u)
    rp_v = view(rp, mod.nvar_u_padded+1:mod.nvar)

    root = 0
    is_root = MPI.Comm_rank(env.comm) == root
    nproc = MPI.Comm_size(env.comm)
    nline_local = div(mod.nline, nproc, RoundUp)

    u_br_root = view(x_curr, 2*mod.ngen+1:mod.nvar_u_padded)
    z_br_root = view(z_curr, 2*mod.ngen+1:mod.nvar_u_padded)
    l_br_root = view(l_curr, 2*mod.ngen+1:mod.nvar_u_padded)

    if env.use_gpu
        u_br_local = CUDA.zeros(Float64, 8*nline_local)
        z_br_local = CUDA.zeros(Float64, 8*nline_local)
        l_br_local = CUDA.zeros(Float64, 8*nline_local)
        xbar_local = CUDA.zeros(Float64, length(xbar_curr))
    else
        u_br_local = zeros(8*nline_local)
        z_br_local = zeros(8*nline_local)
        l_br_local = zeros(8*nline_local)
        xbar_local = zeros(length(xbar_curr))
    end

    MPI.Scatter!(u_br_root, u_br_local, root, env.comm)
    MPI.Scatter!(xbar_curr, xbar_local, root, env.comm)

    nblk_gen = div(mod.ngen, 32, RoundUp)
    nblk_br = mod.nline
    nblk_bus = div(mod.nbus, 32, RoundUp)

    beta = 1e3
    gamma = 6.0 # TODO: not used
    c = 6.0
    theta = 0.8
    sqrt_d = sqrt(mod.nvar_u + mod.nvar_v)
    OUTER_TOL = sqrt_d*(par.outer_eps)

    time_gen = time_br = time_bus = 0
    shift_lines = 0
    shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

    mismatch = Inf
    z_prev_norm = z_curr_norm = Inf

    outer_iter = 0
    while outer_iter < outer_iterlim
        outer_iter += 1

        inner_iter = 0
        while inner_iter < inner_iterlim
            inner_iter += 1
        end # inner while loop

    end # outer while loop
end
