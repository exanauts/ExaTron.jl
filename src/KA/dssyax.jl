@inline function ExaTron.nrm2!(wa, A, n::Int, tx)

    v = 0.0
    if tx <= n
        @inbounds for j=1:n
            v += A[j,tx]^2
        end
        @inbounds wa[tx] = sqrt(v)
    end
    @synchronize

    return
end

@inline function ExaTron.dssyax(n::Int, A,
                        z,
                        q,
                        tx)
    v = 0.0
    if tx <= n
        @inbounds for j=1:n
            v += A[tx,j]*z[j]
        end
        @inbounds q[tx] = v
    end
    @synchronize

    return
end

@inline function ExaTron.reorder!(n::Int, nfree::Int, B,
                          A, indfree,
                          iwa,
                          tx)
    if tx <= nfree
        @inbounds begin
            jfree = indfree[tx]
            B[tx,tx] = A[jfree,jfree]
            for i=jfree+1:n
                if iwa[i] > 0
                    B[iwa[i],tx] = A[i,jfree]
                    B[tx,iwa[i]] = B[iwa[i],tx]
                end
            end
        end
    end

    @synchronize

    return
end
