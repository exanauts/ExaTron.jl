
abstract type AbstractPreconditionner end

function update! end
# L*y = x    &    L'*y = x
# TODO: clean signature
function dstrsol end

# No precond
struct EyePreconditionner <: AbstractPreconditionner end
update!(e::EyePreconditionner, B::TronSparseMatrixCSC, nfree, nnz, iwa, wa1, wa2) = nothing
dstrsol(n, P::EyePreconditionner, r, task) = nothing

# IncompleteCholesky precond
struct IncompleteCholesky <: AbstractPreconditionner
    L::AbstractTronMatrix
    memory::Int
    nv::Int
end
IncompleteCholesky(L::AbstractTronMatrix, p::Int) = IncompleteCholesky(L, p, 5)

# Compute the incomplete Cholesky factorization.
function update!(icfs::IncompleteCholesky, B::AbstractTronMatrix,
                 nfree, nnz, iwa, wa1, wa2)
    T = eltype(wa1)
    alpha = T(0)
    # Compute the incomplete Cholesky factorization.
    ICFS.dicfs(
        nfree, nnz, B, icfs.L,
        icfs.nv, alpha, iwa, wa1, wa2
    )
end

dstrsol(n, P::IncompleteCholesky, r, task) = dstrsol(n, P.L, r, task)

