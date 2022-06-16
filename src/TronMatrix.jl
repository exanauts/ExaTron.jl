tron_zeros(S, n) = fill!(S(undef, Int64(n)), zero(eltype(S)))
tron_zeros(S, dims::Tuple) = fill!(S(undef, Int64(dims[1]), Int64(dims[1])), zero(eltype(S)))

export TronSparseMatrixCSC

abstract type AbstractTronMatrix end

struct TronSparseMatrixCSC{VI, VD} <: AbstractTronMatrix
    n::Int
    nnz::Int
    colptr::VI
    rowval::VI
    map::VI
    diag_vals::VD
    tril_vals::VD
end

mutable struct TronDenseMatrix{MD} <: AbstractTronMatrix
    n::Int      # the current dimension of submatrix
    max_n::Int  # the maximum dimension this matrix can hold
    vals::MD    # matrix entries
end

function TronDenseMatrix{MD}(n::Int) where {MD}
    @assert n >= 1
    vals = tron_zeros(MD, (n, n))
    return TronDenseMatrix{MD}(n, n, vals)
end

function TronDenseMatrix(A::TronDenseMatrix)
    return TronDenseMatrix{typeof(A.vals)}(A.n, A.max_n, copy(A.vals))
end

function TronDenseMatrix(I::VI, J::VI, V::Array, n) where {VI}
    @assert n >= 1
    @assert length(I) == length(J) == length(V)

    A = TronDenseMatrix{Array{Float64, 2}}(n, n, tron_zeros(Array{eltype(V)}, (n, n)))
    for i=1:length(I)
        @assert 1 <= I[i] <= n && 1 <= J[i] <= n && I[i] >= J[i]
        @inbounds A.vals[I[i], J[i]] += V[i]
    end

    return A
end

# Default constructor to allocate memory
function TronSparseMatrixCSC{VI, VD}(n::Int, nnz::Int) where {VI, VD}
    colptr = tron_zeros(VI, n+1)
    rowval = tron_zeros(VI, nnz)
    diag_vals = tron_zeros(VD, n)
    tril_vals = tron_zeros(VD, nnz)
    map = tron_zeros(VI, nnz)
    return TronSparseMatrixCSC{VI, VD}(
        n, nnz, colptr, rowval, map, diag_vals, tril_vals,
    )
end

# Constructor using sparsity pattern passed as input
function TronSparseMatrixCSC(I::VI, J::VI, V::VD, n) where {VI, VD}
    @assert length(I) == length(J) == length(V)
    # Create a CSC matrix without duplicates.
    nnz = length(I)
    # Temporary arrays for removal of duplicates
    csc_rows = zeros(Int, nnz)
    csc_cols = zeros(Int, n+1)
    imap = zeros(Int, nnz)
    w = zeros(Int, n)

    acol_ptr = tron_zeros(VI, n+1)
    arow_ind = tron_zeros(VI, nnz)
    diag_vals = tron_zeros(VD, n)
    tril_vals = tron_zeros(VD, nnz)
    map = tron_zeros(VI, nnz)

    # Count the number of entries of each column.
    for i in 1:nnz
        csc_cols[J[i]+1] += 1
    end

    # Compute the starting address (zero-based) of each column.
    for i in 2:n
        csc_cols[i+1] += csc_cols[i]
    end

    # Construct a CSC matrix.
    for i=1:nnz
        p = csc_cols[J[i]]
        csc_rows[p+1] = I[i]
        imap[p+1] = i
        csc_cols[J[i]] += 1
    end

    # Reset the starting address (one-based) of each column.
    for i=n+1:-1:2
        csc_cols[i] = csc_cols[i-1] + 1
    end
    csc_cols[1] = 1

    # Remove duplicates.
    fill!(w, -1)
    nz = 1
    for i=1:n
        p = nz
        for j=csc_cols[i]:csc_cols[i+1]-1
            r = csc_rows[j]
            if w[r] >= p  # already seen (r,i)
                if i == r # diagonal term
                    map[imap[j]] = -r
                else
                    map[imap[j]] = w[r]
                end
            else # first time seen (r,i)
                w[r] = nz
                if i == r # diagonal term
                    map[imap[j]] = -r
                else
                    map[imap[j]] = nz
                    arow_ind[nz] = r
                    nz = nz + 1
                end
            end
        end
        acol_ptr[i] = p
    end
    acol_ptr[n+1] = nz
    nnz_a = nz - 1

    return TronSparseMatrixCSC{VI, VD}(
        n, nnz_a, acol_ptr, arow_ind, map, diag_vals, tril_vals,
    )
end

Base.size(A::TronDenseMatrix) = (A.n, A.n)
Base.size(A::TronSparseMatrixCSC) = (A.n, A.n)

function Base.fill!(A::TronDenseMatrix, val)
    fill!(A.vals, val)
end

function Base.fill!(A::TronSparseMatrixCSC, val)
    fill!(A.diag_vals, val)
    fill!(A.tril_vals, val)
end

function transfer!(A::TronSparseMatrixCSC, rows, cols, values, nnz)
    for i in 1:nnz
        m = A.map[i]
        if m < 0
            A.diag_vals[-m] = values[i]
        else
            A.tril_vals[m] = values[i]
        end
    end
end
function transfer!(A::TronDenseMatrix, rows, cols, values, nnz)
    @inbounds for i in 1:nnz
        # It is assumed that rows[i] >= cols[i] for all i.
        A.vals[rows[i], cols[i]] += values[i]
    end
end

nrm2!(wa, A) = nrm2!(wa, A, A.n)

function nrm2!(wa, A::TronDenseMatrix, n)
    @inbounds for i=1:n
        wa[i] = A.vals[i,i]^2
    end
    @inbounds for j=1:n
        for i=j+1:n
            wa[j] += A.vals[i,j]^2
            wa[i] += A.vals[i,j]^2
        end
    end
    @inbounds for j=1:n
        wa[j] = sqrt(wa[j])
    end

    return
end

function nrm2!(wa, A::TronSparseMatrixCSC, n)
    @inbounds for i=1:n
        wa[i] = A.diag_vals[i]^2
    end
    @inbounds for j=1:n
        for i=A.colptr[j]:A.colptr[j+1]-1
            k = A.rowval[i]
            wa[j] = wa[j] + A.tril_vals[i]^2
            wa[k] = wa[k] + A.tril_vals[i]^2
        end
    end
    @inbounds for j=1:n
        wa[j] = sqrt(wa[j])
    end
end