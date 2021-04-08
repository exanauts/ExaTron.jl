mutable struct ExaTronProblem{VI, VD, TM <: AbstractTronMatrix}
    n::Cint                 # number of variables
    nnz::Integer            # number of Hessian entries
    nnz_a::Integer          # number of Hessian entries in the strict lower
    A::TM
    B::TM
    L::TM
    indfree::VI   # a working array of dimension n
    iwa::VI       # a working array of dimension 3*n
    g::VD         # gradient
    xc::VD        # a working array of dimension n
    s::VD         # a working array of dimension n
    wa::VD        # a working array of dimension 7*n

    x::VD
    x_l::VD
    x_u::VD
    rows::VI
    cols::VI
    values::VD

    eval_f::Function
    eval_grad_f::Function
    eval_h::Function

    options::Dict{String, Any}

    # Statistics
    delta::Cdouble
    f::Cdouble
    gnorm_two::Float64
    gnorm_inf::Float64
    nfev::Integer
    ngev::Integer
    nhev::Integer
    minor_iter::Integer
    status::Symbol

    ExaTronProblem{VI, VD, TM}() where {VI, VD, TM} = new{VI, VD, TM}()
end

function gpnorm(n, x, x_l, x_u, g)
    T = eltype(x)
    two_norm = zero(T)
    inf_norm = zero(T)

    for i=1:n
        if x_l[i] != x_u[i]
            if x[i] == x_l[i]
                val = (min(g[i], zero(T)))^2
            elseif x[i] == x_u[i]
                val = (max(g[i], zero(T)))^2
            else
                val = g[i]^2
            end

            two_norm += val
            val = sqrt(val)
            if inf_norm < val
                inf_norm = val
            end
        end
    end

    two_norm = sqrt(two_norm)

    return two_norm, inf_norm
end

function set_default_options!(prob::ExaTronProblem)
    prob.options = Dict{String,Any}()
    set_default_options!(prob.options)
    return
end

function set_default_options!(options::Dict{String, Any})
    options["max_feval"] = 500
    options["max_minor"] = 200
    options["p"] = 5
    options["verbose"] = 0
    options["tol"] = 1e-6
    options["fatol"] = 0.0
    options["frtol"] = 1e-12
    options["fmin"] = -1e32
    options["cgtol"] = 0.1
    options["tron_code"] = :Julia
    options["matrix_type"] = :Sparse

    return
end

function getOption(prob::ExaTronProblem, keyword::String)
    if !haskey(prob.options, keyword)
        error("ExaTron does not have option with name $(keyword).")
    end
    return prob.options[keyword]
end

function setOption(prob::ExaTronProblem, keyword::String, value::Symbol)
    if !haskey(prob.options, keyword)
        error("ExaTron does not have option with name $(keyword).")
    end
    prob.options[keyword] = value
end

function setOption(prob::ExaTronProblem, keyword::String, value::Integer)
    if !haskey(prob.options, keyword)
        error("ExaTron does not have option with name $(keyword).")
    end
    prob.options[keyword] = value
    return
end

function setOption(prob::ExaTronProblem, keyword::String, value::Float64)
    if !haskey(prob.options, keyword)
        error("ExaTron does not have option with name $(keyword).")
    end
    prob.options[keyword] = value
    return
end

function instantiate_memory!(tron::ExaTronProblem{VI,VD}, n, nele_hess) where {VI,VD}
    tron.n = convert(Cint, n)
    tron.nnz = nele_hess
    tron.indfree = VI(undef, n)
    tron.iwa = VI(undef, 3*n)

    tron.g = tron_zeros(VD, n)
    tron.xc = tron_zeros(VD, n)
    tron.s = tron_zeros(VD, n)
    tron.wa = tron_zeros(VD, 7*n)

    tron.x = tron_zeros(VD, n)
    tron.x_l = tron_zeros(VD, n)
    tron.x_u = tron_zeros(VD, n)
    tron.rows = VI(undef, nele_hess)
    tron.cols = VI(undef, nele_hess)
    tron.values = tron_zeros(VD, nele_hess)
end

function createProblem(n::Integer, x_l::AbstractVector{T}, x_u::AbstractVector{T},
                       nele_hess::Integer, eval_f_cb, eval_grad_f_cb, eval_h_cb; options...) where T
    @assert n == length(x_l) == length(x_u)
    @assert typeof(x_l) == typeof(x_u)

    # Load options first, as we need to know :matrix_type for type inference
    options_dict = Dict{String, Any}()
    set_default_options!(options_dict)
    for (name, value) in options
        options_dict[string(name)] = value
    end

    VI = Vector{Cint}
    VD = typeof(x_l)

    if options_dict["matrix_type"] == :Sparse
        TM = TronSparseMatrixCSC{VI, VD}
    else
        AT = isa(x_l, Array) ? Array{T, 2} : CuArray{T, 2}
        TM = TronDenseMatrix{AT}
    end

    tron = ExaTronProblem{VI, VD, TM}()
    tron.options = options_dict
    instantiate_memory!(tron, n, Int64(nele_hess))
    copyto!(tron.x_l, 1, x_l, 1, n)
    copyto!(tron.x_u, 1, x_u, 1, n)

    tron.eval_f = eval_f_cb
    tron.eval_grad_f = eval_grad_f_cb
    tron.eval_h = eval_h_cb

    tron.eval_h(tron.x, :Structure, tron.rows, tron.cols, 1.0, Float64[], Float64[])
    # Instantiate sparse matrix
    p = tron.options["p"]

    if tron.options["matrix_type"] == :Sparse
        tron.A = TronSparseMatrixCSC(tron.rows, tron.cols, tron.values, n)
        tron.B = TronSparseMatrixCSC{VI, VD}(n, Int64(nele_hess))
        tron.L = TronSparseMatrixCSC{VI, VD}(n, Int64(nele_hess + n*p))
        tron.nnz_a = tron.A.nnz
    else
        tron.A = TronDenseMatrix(tron.rows, tron.cols, tron.values, n)
        if isa(x_l, Array)
            tron.B = TronDenseMatrix{Array{T, 2}}(n)
            tron.L = TronDenseMatrix{Array{T, 2}}(n)
        else
            tron.B = TronDenseMatrix{CuArray{T, 2}}(n)
            tron.L = TronDenseMatrix{CuArray{T, 2}}(n)
        end
        tron.nnz_a = n*n
    end
    tron.status = :NotSolved

    return tron
end

function solveProblem(tron::ExaTronProblem)
    task = Vector{UInt8}(undef, 60)
    for (i,s) in enumerate("START")
        task[i] = UInt8(s)
    end

    VI = Vector{Cint}
    VD = typeof(tron.x)
    T = eltype(tron.x)

    isave = VI(undef, 3)
    dsave = tron_zeros(VD, 3)
    fatol = T(tron.options["fatol"])
    frtol = T(tron.options["frtol"])
    fmin = T(tron.options["fmin"])
    cgtol = T(tron.options["cgtol"])
    gtol = T(tron.options["tol"])
    max_feval = tron.options["max_feval"]::Int
    tcode = tron.options["tron_code"]::Symbol
    max_minor = tron.options["max_minor"]::Int

    # Project x into its bound. Tron requires x to be feasible.
    tron.x .= max.(tron.x_l, min.(tron.x_u, tron.x))

    itermax = tron.n
    tron.minor_iter = 0
    tron.nfev = tron.ngev = tron.nhev = 0
    tron.status = :NotSolved
    search = true

    while (search)

        # Evaluate the function.

        if Char(task[1]) == 'F' || unsafe_string(pointer(task), 5) == "START"
            tron.f = tron.eval_f(tron.x)
            tron.nfev += 1
            if tron.nfev >= max_feval
                tron.status = :Maximum_Iterations_Exceeded
                search = false
            end
        end

        # Evaluate the gradient and Hessian.

        if (Char(task[1]) == 'G' && Char(task[2]) == 'H') || unsafe_string(pointer(task), 5) == "START"
            tron.eval_grad_f(tron.x, tron.g)
            tron.eval_h(tron.x, :Values, Int[], Int[], T(1.0), T[], tron.values)
            tron.ngev += 1
            tron.nhev += 1
            tron.minor_iter += 1

            # Copy values in the CSC matrix.
            fill!(tron.A, T(0.0))
            transfer!(tron.A, tron.rows, tron.cols, tron.values, tron.nnz)
        end

        # Initialize the trust region bound.

        if unsafe_string(pointer(task), 5) == "START"
            gnorm0 = norm(tron.g)
            tron.delta = gnorm0
        end

        # Call Tron.

        if search
            if tcode == :Fortran
                delta = Ref{Cdouble}(tron.delta)
                # TODO
                ccall((:dtron_, EXATRON_LIBRARY),
                    Cvoid,
                    (Ref{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                    Ref{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                    Ptr{Cint}, Ptr{Cint}, Ref{Cdouble}, Ref{Cdouble},
                    Ref{Cdouble}, Ref{Cdouble}, Ref{Cint}, Ref{Cdouble},
                    Ptr{UInt8}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}),
                    tron.n, tron.x, tron.x_l, tron.x_u,
                    tron.f, tron.g, tron.A.tril_vals, tron.A.diag_vals,
                    tron.A.colptr, tron.A.rowval, frtol, fatol,
                    fmin, cgtol, itermax, delta,
                    task, tron.B.tril_vals, tron.B.diag_vals, tron.B.colptr,
                    tron.B.rowval, tron.L.tril_vals, tron.L.diag_vals, tron.L.colptr,
                    tron.L.rowval, tron.xc, tron.s, tron.indfree,
                    isave, dsave, tron.wa, tron.iwa)
                tron.delta = delta[]
            else
                tron.delta = ExaTron.dtron(tron.n, tron.x, tron.x_l, tron.x_u,
                                tron.f, tron.g, tron.A,
                                frtol, fatol,
                                fmin, cgtol, itermax, tron.delta,
                                task, tron.B, tron.L,
                                tron.xc, tron.s, tron.indfree,
                                isave, dsave, tron.wa, tron.iwa)
            end
        end

        if unsafe_string(pointer(task), 4) == "NEWX"
            tron.gnorm_two, tron.gnorm_inf = gpnorm(tron.n, tron.x, tron.x_l, tron.x_u, tron.g)
            if tron.gnorm_inf <= gtol
                for (i,s) in enumerate("CONVERGENCE: GTOL TEST SATISFIED")
                    task[i] = UInt8(s)
                end
            end

            if tron.minor_iter >= max_minor
                tron.status = :Maximum_Iterations_Exceeded
                search = false
            end
        end

        if unsafe_string(pointer(task), 4) == "CONV"
            tron.status = :Solve_Succeeded
            search = false
        end

        if unsafe_string(pointer(task), 4) == "WARN"
            tron.status = :TooSmall
            search = false
        end
    end

    return tron.status
end
