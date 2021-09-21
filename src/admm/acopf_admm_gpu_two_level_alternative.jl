"""
    admm_restart

This function restarts the ADMM with a given `env::AdmmEnv` containing solutions and all the other parameters.
"""
function admm_restart_two_level_alternative(env::AdmmEnv, mod::Model; outer_iterlim=10, inner_iterlim=800, scale=1e-4)
    data, par, sol = env.data, env.params, mod.solution

    shift_lines = 0
    shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

    nblk_gen = div(mod.ngen, 32, RoundUp)
    nblk_br = mod.nline
    nblk_bus = div(mod.nbus, 32, RoundUp)

    it = 0
    time_gen = time_br = time_bus = 0.0

    beta = 1e3
    c = 6.0
    theta = 0.8
    sqrt_d = sqrt(mod.nvar)
    OUTER_TOL = sqrt_d*(par.outer_eps)

    outer = inner = cumul = 0
    mismatch = Inf
    z_prev_norm = z_curr_norm = Inf

    overall_time = @timed begin
    while outer < outer_iterlim
        outer += 1

        if !env.use_gpu
            sol.z_outer .= sol.z_curr
            z_prev_norm = norm(sol.z_outer)
        else
            CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1,64)+1) copy_data_kernel(mod.nvar, sol.z_outer, sol.z_curr)
            z_prev_norm = CUDA.norm(sol.z_curr)
        end

        inner = 0
        while inner < inner_iterlim
            inner += 1
            cumul += 1

            if !env.use_gpu
                sol.z_prev .= sol.z_curr

                tcpu = generator_kernel_two_level(mod, mod.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)
                time_gen += tcpu.time

                if env.use_linelimit
                    tcpu = @timed auglag_it, tron_it = auglag_linelimit_two_level_alternative(mod.n, mod.nline, mod.line_start,
                                                        inner, par.max_auglag, par.mu_max, scale,
                                                        sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                        shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                        mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
                else
                    tcpu = @timed auglag_it, tron_it = polar_kernel_two_level_alternative(mod.n, mod.nline, mod.line_start, scale,
                                                            sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                            shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                            mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
                end
                time_br += tcpu.time

                tcpu = @timed bus_kernel_two_level_alternative(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start,
                                            mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                                            mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr, sol.z_curr,
                                            sol.l_curr, sol.rho, mod.YshR, mod.YshI)
                time_bus += tcpu.time

                sol.z_curr .= (-(sol.lz .+ sol.l_curr .+ sol.rho.*(sol.u_curr .- sol.v_curr))) ./ (beta .+ sol.rho)
                sol.l_curr .= -(sol.lz .+ beta.*sol.z_curr)
                sol.rp .= sol.u_curr .- sol.v_curr .+ sol.z_curr
                sol.rd .= sol.z_curr .- sol.z_prev
                sol.Ax_plus_By .= sol.rp .- sol.z_curr

                primres = norm(sol.rp)
                dualres = norm(sol.rd)
                z_curr_norm = norm(sol.z_curr)
                mismatch = norm(sol.Ax_plus_By)

                eps_pri = sqrt_d / (2500*outer)
                #=
                eps_pri = sqrt(length(sol.l_curr))*par.ABSTOL + par.RELTOL*max(norm(sol.u_curr), norm(-sol.v_curr))
                eps_dual = sqrt(length(sol.u_curr))*par.ABSTOL + par.RELTOL*norm(sol.l_curr)
                =#

                if par.verbose > 0
                    if inner == 1 || (inner % 50) == 0
                        @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                        "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "OuterTol", "Beta")
                    end

                    @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                            outer, inner, primres, eps_pri, dualres, z_curr_norm, mismatch, OUTER_TOL, beta)
                end

                if primres <= eps_pri || dualres <= par.DUAL_TOL
                    break
                end
            else
                CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.z_prev, sol.z_curr)

                tgpu = generator_kernel_two_level(mod, mod.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)
                time_gen += tgpu.time

                if env.use_linelimit
                    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size auglag_linelimit_two_level_alternative(
                                                        mod.n, mod.nline, mod.line_start,
                                                        inner, par.max_auglag, par.mu_max, scale,
                                                        sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                        shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                        mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
                else
                    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size polar_kernel_two_level_alternative(mod.n, mod.nline, mod.line_start, scale,
                                                                    sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                                    shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                                    mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrBound, mod.ToBound)
                end
                time_br += tgpu.time
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel_two_level_alternative(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start,
                                                                            mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                                                                            mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr,
                                                                            sol.z_curr, sol.l_curr, sol.rho, mod.YshR, mod.YshI)
                time_bus += tgpu.time

                CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_zv_kernel(mod.nvar, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho, sol.lz, beta)
                @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_l_kernel(mod.nvar, sol.l_curr, sol.z_curr, sol.lz, beta)
                @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) compute_primal_residual_v_kernel(mod.nvar, sol.rp, sol.u_curr, sol.v_curr, sol.z_curr)
                @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, sol.rd, sol.z_curr, sol.z_prev)
                CUDA.synchronize()

                CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, sol.Ax_plus_By, sol.rp, sol.z_curr)

                mismatch = CUDA.norm(sol.Ax_plus_By)
                primres = CUDA.norm(sol.rp)
                dualres = CUDA.norm(sol.rd)
                z_curr_norm = CUDA.norm(sol.z_curr)
                eps_pri = sqrt_d / (2500*outer)

                #gpu_eps_pri = sqrt(length(sol.l_curr))*par.ABSTOL + par.RELTOL*max(CUDA.norm(sol.u_curr), CUDA.norm(sol.v_curr))
                #gpu_eps_dual = sqrt(length(sol.u_curr))*par.ABSTOL + par.RELTOL*CUDA.norm(sol.l_curr)

                if par.verbose > 0
                    if inner == 1 || (inner % 50) == 0
                        @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                        "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||", "||Ax+By||", "OuterTol", "Beta")
                    end

                    @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                            outer, inner, primres, eps_pri, dualres, z_curr_norm, mismatch, OUTER_TOL, beta)
                end

                if primres <= eps_pri || dualres <= par.DUAL_TOL
                    break
                end
            end
        end # while inner

        if mismatch <= OUTER_TOL
            break
        end

        if !env.use_gpu
            sol.lz .= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, sol.lz .+ (beta .* sol.z_curr)))
        else
            CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_lz_kernel(mod.nvar, par.MAX_MULTIPLIER, sol.z_curr, sol.lz, beta)
        end

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end # while outer
    end # @timed

    u_curr = zeros(mod.nvar)
    v_curr = zeros(mod.nvar)
    copyto!(u_curr, sol.u_curr)
    copyto!(v_curr, sol.v_curr)
    objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)])^2 +
                 data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)]) +
                 data.generators[g].coeff[data.generators[g].n]
                 for g in 1:mod.ngen)::Float64
    sol.objval = objval

    if par.verbose > 0
        pg_err, qg_err = check_generator_bounds(mod, u_curr)
        vm_err = check_voltage_bounds_alternative(mod, u_curr)
        real_err, reactive_err = check_power_balance_alternative(mod, u_curr)
        u_objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)])^2 +
                       data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)]) +
                       data.generators[g].coeff[data.generators[g].n]
                       for g in 1:mod.ngen)::Float64
        @printf(" ** Violations of u_curr\n")
        @printf("Real power generator bounds     = %.6e\n", pg_err)
        @printf("Reactive power generator bounds = %.6e\n", qg_err)
        @printf("Voltage bounds                  = %.6e\n", vm_err)
        @printf("Real power balance              = %.6e\n", real_err)
        @printf("Reactive power balance          = %.6e\n", reactive_err)
        @printf("Objective value                 = %.6e\n", u_objval)

        pg_err, qg_err = check_generator_bounds(mod, v_curr)
        vm_err = check_voltage_bounds_alternative(mod, v_curr)
        real_err, reactive_err = check_power_balance_alternative(mod, v_curr)
        v_objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*v_curr[mod.gen_start+2*(g-1)])^2 +
                       data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*v_curr[mod.gen_start+2*(g-1)]) +
                       data.generators[g].coeff[data.generators[g].n]
                       for g in 1:mod.ngen)::Float64
        @printf(" ** Violations of v_curr\n")
        @printf("Real power generator bounds     = %.6e\n", pg_err)
        @printf("Reactive power generator bounds = %.6e\n", qg_err)
        @printf("Voltage bounds                  = %.6e\n", vm_err)
        @printf("Real power balance              = %.6e\n", real_err)
        @printf("Reactive power balance          = %.6e\n", reactive_err)
        @printf("Objective value                 = %.6e\n", v_objval)

        pg_err, qg_err = check_generator_bounds(mod, u_curr)
        vm_err = check_voltage_bounds_alternative(mod, v_curr)
        real_err, reactive_err = check_power_balance_alternative(mod, u_curr, v_curr)
        @printf(" ** Violations of (u_curr, v_curr)\n")
        @printf("Real power generator bounds     = %.6e\n", pg_err)
        @printf("Reactive power generator bounds = %.6e\n", qg_err)
        @printf("Voltage bounds                  = %.6e\n", vm_err)
        @printf("Real power balance              = %.6e\n", real_err)
        @printf("Reactive power balance          = %.6e\n", reactive_err)
        @printf("Objective value abs(v_objval-u_objval)/u_objval = %.6e\n", abs(v_objval-u_objval)/u_objval)

        rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol = check_linelimit_violation(data, u_curr)
        @printf(" ** Line limit violations of u_curr\n")
        @printf("RateA number of violations = %d (%d)\n", rateA_nviols, mod.nline)
        @printf("RateA maximum violation    = %.6e\n", rateA_maxviol)
        @printf("RateC number of violations = %d (%d)\n", rateC_nviols, mod.nline)
        @printf("RateC maximum violation    = %.6e\n", rateC_maxviol)

        rateA_nviols, rateA_maxviol, rateC_nviols, rateC_maxviol = check_linelimit_violation(data, v_curr)
        @printf(" ** Line limit violations of v_curr\n")
        @printf("RateA number of violations = %d (%d)\n", rateA_nviols, mod.nline)
        @printf("RateA maximum violation    = %.6e\n", rateA_maxviol)
        @printf("RateC number of violations = %d (%d)\n", rateC_nviols, mod.nline)
        @printf("RateC maximum violation    = %.6e\n", rateC_maxviol)

        @printf(" ** Statistics\n")
        @printf("Outer iterations . . . . . . . . . %5d\n", outer)
        @printf("Cumulative iterations  . . . . . . %5d\n", cumul)
        @printf("Time per iteration . . . . . . . . %5.2f (secs/iter)\n", overall_time.time / cumul)
        @printf("Overall time . . . . . . . . . . . %5.2f (secs)\n", overall_time.time)
        @printf("Generator time . . . . . . . . . . %5.2f (secs)\n", time_gen)
        @printf("Branch time. . . . . . . . . . . . %5.2f (secs)\n", time_br)
        @printf("Bus time . . . . . . . . . . . . . %5.2f (secs)\n", time_bus)
        @printf("G+Br+B time. . . . . . . . . . . . %5.2f (secs)\n", time_gen + time_br + time_bus)
    end
    return
end

function admm_rect_gpu_two_level_alternative(case::String;
    case_format="matpower",
    outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
    use_gpu=false, use_linelimit=false, outer_eps=2*1e-4, solve_pf=false, gpu_no=0, verbose=1)

    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
            use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=false,
            solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose)
    env.params.outer_eps = outer_eps
    mod = Model{T,TD,TI,TM}(env)

    if use_gpu
        # Set rateA in membuf.
        CUDA.@sync @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) set_rateA_kernel(mod.nline, mod.membuf, mod.rateA)
    else
        mod.membuf[29,:] .= mod.rateA
    end

    admm_restart_two_level_alternative(env, mod; outer_iterlim=outer_iterlim, inner_iterlim=inner_iterlim, scale=scale)
    return env, mod
end
