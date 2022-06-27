function ExaTron.TronDenseMatrix(I::VI, J::VI, V::CuArray, n) where {VI <: CuVector{Int}}
    @assert n >= 1
    @assert length(I) == length(J) == length(V)

    A = TronDenseMatrix{CuArray{Float64, 2}}(n, n, tron_zeros(CuArray{eltype(V)}, (n, n)))
    for i=1:length(I)
        @assert 1 <= I[i] <= n && 1 <= J[i] <= n && I[i] >= J[i]
        @inbounds A.vals[I[i], J[i]] += V[i]
    end

    return A
end

@inline function Base.fill!(w::CuDeviceArray{Float64,1}, val::Float64)
    tx = threadIdx().x
    bx = blockIdx().x

    if tx <= length(w) && bx == 1
        @inbounds w[tx] = val
    end
    CUDA.sync_threads()

    return
end
