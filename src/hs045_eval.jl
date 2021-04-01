# Original HS45 problem is defined for n=5 and as follows:
#
# minimize    2.0 - (prod_{i=1}^n x[i]) / 120
# subject to  x[i] <= i, for i=1..n
#
# The optimal solution is x[i] = i for i=1..n.
# We extended it to take n in [1,32].

function eval_f_kernel(n, x)
    v = 1.0
    @inbounds for i=1:n
        v *= x[i]
    end
    f = 2.0 - v/120
    CUDA.sync_threads()
    return f
end

function eval_grad_f_kernel(n, x, g)
    tx = threadIdx().x
    ty = threadIdx().y

    if tx == 1 && ty == 1
        @inbounds for j=1:n
            v = 1.0
            @inbounds for i=1:n
                if i != j
                    v *= x[i]
                end
            end
            g[j] = -v/120
        end
    end
    CUDA.sync_threads()
    return
end

function eval_h_kernel(n, x, A)
    tx = threadIdx().x
    ty = threadIdx().y

    if tx == 1 && ty == 1
        v = 1.0
        @inbounds for j=1:n
            v *= x[j]
        end

        @inbounds for j=1:n
            A[j,j] = 0.0
            @inbounds for i=j+1:n
                A[i,j] = -( (v / (x[j]*x[i])) / 120 )
            end
        end
    end

    CUDA.sync_threads()
    if tx <= n && ty == 1
        @inbounds for j=1:n
            if tx > j
                A[j,tx] = A[tx,j]
            end
        end
    end
    CUDA.sync_threads()

    return
end

function eval_f_cpu(x)
    n = length(x)
    v = 1.0
    @inbounds for i=1:n
        v *= x[i]
    end
    f = 2.0 - v/120

    return f
end

function eval_grad_f_cpu(x, g)
    n = length(x)
    @inbounds for j=1:n
        v = 1.0
        @inbounds for i=1:n
            if i != j
                v *= x[i]
            end
        end
        g[j] = -v/120
    end

    return
end

function eval_h_cpu(x, mode, rows, cols, scale, lambda, values)
    n = length(x)

    if mode == :Structure
        nz = 1
        @inbounds for j=1:n
            rows[nz] = j; cols[nz] = j; nz += 1
            @inbounds for i=j+1:n
                rows[nz] = i; cols[nz] = j; nz += 1
            end
        end
    else
        v = 1.0
        @inbounds for j=1:n
            v *= x[j]
        end

        nz = 1
        @inbounds for j=1:n
            values[nz] = 0.0; nz += 1
            @inbounds for i=j+1:n
                values[nz] = -( (v / (x[j]*x[i])) / 120 )
                nz += 1
            end
        end
    end

    return
end