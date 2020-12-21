
"""
    AbstractPreconditionner

Abstract supertype to implement a preconditioner for
conjugate gradient.
"""
abstract type AbstractPreconditionner end

"""
    update!(P::AbstractPreconditionner, B::AbstractTronMatrix, nfree, nnz, iwa, wa1, wa2)

Update coefficients of preconditioner according to the new matrix `B`.
`nfree` is the degree of freedom, and `nnz` the current number of
elements in `B`.

"""
function update! end

"""
    dstrsol(n, P::AbstractPreconditionner, r, task)

Apply preconditioner on the `n` first components of the
current vector `r`.
If `task` is set equal to `N`, the preconditioner is applied as is.
If `task` is equal to `T`, then the transpose of the preconditioner operator is applied.

"""
function dstrsol end


"""
    EyePreconditionner <: AbstractPreconditionner

Apply identity as preconditioning. Equivalent to no preconditioning.

"""
struct EyePreconditionner <: AbstractPreconditionner end

update!(e::EyePreconditionner, B::AbstractTronMatrix, nfree, nnz, iwa, wa1, wa2) = nothing

dstrsol(n, P::EyePreconditionner, r, task) = nothing

"""
    IncompleteCholesky <: AbstractPreconditionner

Use an Incomplete Cholesky Factorization routine as a preconditioner.
The incomplete factorization is computed when calling the function
`update!`, and the factorization stored inside the object `IncompleteCholesky`.

"""
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
    ICFS.dicfs(
        nfree, nnz, B, icfs.L,
        icfs.nv, alpha, iwa, wa1, wa2
    )
end

dstrsol(n, P::IncompleteCholesky, r, task) = dstrsol(n, P.L, r, task)

