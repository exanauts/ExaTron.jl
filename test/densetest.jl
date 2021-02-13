@testset "DenseMatrix" begin
    tol = 1e-8

    Random.seed!(0)
    for k=1:10
        n = 10
        A = ExaTron.TronDenseMatrix{Array}(n)
        B = zeros(n,n)
        for j=1:n, i=j:n
            v = rand(1)[1]
            A.vals[i,j] = v
            B[i,j] = v
            B[j,i] = v
        end

        # Test A == B.
        @test (A.vals == tril(B))

        # Test dssyax().
        x = rand(n)
        y = zeros(n)
        ExaTron.dssyax(n, A, x, y)
        @test norm(y .- (B*x)) <= tol

        # Test nrm2!().
        wa = zeros(n)
        wb = [norm(B[:,i]) for i=1:n]
        ExaTron.nrm2!(wa, A, n)
        @test norm(wa .- wb) <= tol

        # Test reorder!().
        iwa = zeros(Int, n)
        nfree = rand(1:n)[1]
        indfree = sort(StatsBase.sample(1:n, nfree, replace=false))
        for j=1:nfree
            iwa[indfree[j]] = j
        end
        C = ExaTron.TronDenseMatrix{Array}(n)
        ExaTron.reorder!(C, A, indfree, nfree, iwa)
        @test C.vals[1:nfree,1:nfree] == A.vals[indfree,indfree]

        # Test copy constructor.
        L = ExaTron.TronDenseMatrix(A)
        @test L.vals == A.vals

        # Test dnsol() and dtsol().
        fill!(x, 1.0)
        fill!(y, 1.0)
        ExaTron.dnsol(n, L, y)
        @test norm(y .- (A.vals\x)) <= tol
        fill!(x, 1.0)
        fill!(y, 1.0)
        ExaTron.dtsol(n, L, y)
        @test norm(y .- (transpose(A.vals)\x)) <= tol

        # Test dicf().
        A.vals .= tril(L.vals * transpose(L.vals))
        info = ExaTron.dicf(n, 1, A, 1, [], [], [], [])
        @test info == 0
        @test norm(A.vals .- L.vals) <= tol

        # Test Tron.
        A.vals = (L.vals * transpose(L.vals))
        g = ones(n)
        x = zeros(n)
        xc = zeros(n)
        xl = -Inf*ones(n)
        xu = Inf*ones(n)
        eval_f_cb(x) = 0.5*(transpose(x)*A.vals*x) + transpose(g)*x
        function eval_grad_f_cb(x, grad_f)
            grad_f .= A.vals*x .+ g
        end
        function eval_h_cb(x, mode, rows, cols, scale, lambda, values)
            if mode == :Structure
                nz = 1
                for j=1:n,i=j:n
                    rows[nz] = i
                    cols[nz] = j
                    nz += 1
                end
            else
                nz = 1
                for j=1:n,i=j:n
                    values[nz] = A.vals[i,j]
                    nz += 1
                end
            end
        end
        prob = ExaTron.createProblem(n, xl, xu, Int(floor((n*(n+1))/2)),
                                     eval_f_cb, eval_grad_f_cb, eval_h_cb; :matrix_type=>:Dense)
        status = ExaTron.solveProblem(prob)
        @test status == :Solve_Succeeded
        # Default termination tolerance of Tron is 1e-6.
        @test norm(A.vals*prob.x .+ g) <= 1e-6
    end
end