# GPU implementation of generator update of ADMM.
@kernel function generator_kernel(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, v,
    l, rho,
    pgmin, pgmax,
    qgmin, qgmax,
    c2, c1
)
    I_ = @index(Group, Linear)
    J_ = @index(Local, Linear)
    I = J_ + (@groupsize()[1] * (I_ - 1))
    if (I <= ngen)
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] -
                            rho[pg_idx]*v[pg_idx])) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] - rho[qg_idx]*v[qg_idx])) / rho[qg_idx]))
    end
end

function generator_kernel(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, v, l, rho,
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1},
    qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    c2::Array{Float64,1}, c1::Array{Float64,1}
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] -
                               rho[pg_idx]*v[pg_idx])) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] - rho[qg_idx]*v[qg_idx])) / rho[qg_idx]))
    end

    return
end

function generator_kernel(
    gen_mod::GeneratorModel,
    baseMVA::Float64, u, v, l, rho,
    device
)
    nblk = div(gen_mod.ngen, 32, RoundUp)
    nblk
    generator_kernel(device, 32, gen_mod.ngen)(
        baseMVA, gen_mod.ngen, gen_mod.gen_start,
        u, v, l, rho, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax, gen_mod.c2, gen_mod.c1
    )
    KA.synchronize(device)

    return 0.0
end

function generator_kernel(
    gen_mod::GeneratorModel,
    baseMVA::Float64, u, v, l, rho,
    device::KA.CPU
)
    tcpu = @timed generator_kernel(baseMVA, gen_mod.ngen, gen_mod.gen_start,
                u, v, l, rho, gen_mod.pgmin, gen_mod.pgmax, gen_mod.qgmin, gen_mod.qgmax, gen_mod.c2, gen_mod.c1)
    return tcpu
end
