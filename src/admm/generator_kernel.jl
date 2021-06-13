# GPU implementation of generator update of ADMM.
function generator_kernel(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
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

    return
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
    model::Model{Float64,CuArray{Float64,1},CuArray{Int,1}},
    baseMVA::Float64, u, v, l, rho
)
    nblk = div(model.ngen, 32, RoundUp)
    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk generator_kernel(baseMVA, model.ngen, model.gen_start,
                u, v, l, rho, model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1)
    return tgpu
end

function generator_kernel(
    model::Model{Float64,Array{Float64,1},Array{Int,1}},
    baseMVA::Float64, u, v, l, rho
)
    tcpu = @timed generator_kernel(baseMVA, model.ngen, model.gen_start,
                u, v, l, rho, model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1)
    return tcpu
end


function generator_kernel_two_level(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, x::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if (I <= ngen)
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx]))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end

    return
end

function generator_kernel_two_level(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, x, z, l, rho,
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1},
    qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    c2::Array{Float64,1}, c1::Array{Float64,1}
)
    for I=1:ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx]))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end

    return
end

function generator_kernel_two_level(
    model::Model{Float64,CuArray{Float64,1},CuArray{Int,1}},
    baseMVA::Float64, u::CuArray{Float64,1}, xbar::CuArray{Float64,1},
    zu::CuArray{Float64,1}, lu::CuArray{Float64,1}, rho_u::CuArray{Float64,1}
)
    nblk = div(model.ngen, 32, RoundUp)
    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk generator_kernel_two_level(baseMVA, model.ngen, model.gen_start,
                u, xbar, zu, lu, rho_u, model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1)
    return tgpu
end

function generator_kernel_two_level(
    model::Model{Float64,Array{Float64,1},Array{Int,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u
)
    tcpu = @timed generator_kernel_two_level(baseMVA, model.ngen, model.gen_start,
                u, xbar, zu, lu, rho_u, model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1)
    return tcpu
end
