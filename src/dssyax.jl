@inline function nrm2!(wa, A, n)
    tx = threadIdx().x
    ty = threadIdx().y

    v = 0.0
    if tx <= n && ty == 1
        @inbounds for j=1:n
            v += A[j,tx]^2
        end
        @inbounds wa[tx] = sqrt(v)
    end
    CUDA.sync_threads()

    return
end

@inline function dssyax(n, A, z, q)
    tx = threadIdx().x
    ty = threadIdx().y

    v = 0.0
    if tx <= n && ty == 1
        @inbounds for j=1:n
            v += A[tx,j]*z[j]
        end
        @inbounds q[tx] = v
    end
    #=
    v = 0.0
    if tx <= n && ty <= n
        v = A[ty,tx]*z[tx]
    end

    # Sum over the x-dimension: v = sum_tx A[ty,tx]*z[tx].
    # The thread with tx=1 will have the sum in v.

    offset = div(blockDim().x, 2)
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset = div(offset, 2)
    end

    if tx == 1
        q[ty] = v
    end
    =#
    CUDA.sync_threads()

    return
end

@inline function reorder!(n, nfree, B, A, indfree, iwa)
    tx = threadIdx().x
    ty = threadIdx().y

    #=
    if tx == 1 && ty == 1
        @inbounds for j=1:nfree
            jfree = indfree[j]
            B[j,j] = A[jfree,jfree]
            for i=jfree+1:n
                if iwa[i] > 0
                    B[iwa[i],j] = A[i,jfree]
                    B[j,iwa[i]] = A[i,jfree]
                end
            end
        end
    end
    =#

    if tx <= nfree && ty == 1
        @inbounds begin
            jfree = indfree[tx]
            B[tx,tx] = A[jfree,jfree]
            for i=jfree+1:n
                if iwa[i] > 0
                    B[iwa[i],tx] = A[i,jfree]
                    B[tx,iwa[i]] = A[i,jfree]
                end
            end
        end
    end

    CUDA.sync_threads()

    return
end
