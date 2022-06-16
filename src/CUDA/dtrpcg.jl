@inline function ExaTron.dtrpcg(n::Int, A::CuDeviceArray{Float64,2},
                        g::CuDeviceArray{Float64,1}, delta::Float64,
                        L::CuDeviceArray{Float64,2},
                        tol::Float64, stol::Float64, itermax::Int,
                        w::CuDeviceArray{Float64,1},
                        p::CuDeviceArray{Float64,1},
                        q::CuDeviceArray{Float64,1},
                        r::CuDeviceArray{Float64,1},
                        t::CuDeviceArray{Float64,1},
                        z::CuDeviceArray{Float64,1})
  zero = 0.0
  one = 1.0

  # Initialize the iterate w and the residual r.
  fill!(w, 0)

  # Initialize the residual t of grad q to -g.
  # Initialize the residual r of grad Q by solving L*r = -g.
  # Note that t = L*r.
  dcopy(n,g,1,t,1)
  dscal(n,-one,t,1)
  dcopy(n,t,1,r,1)
  dnsol(n, L, r)

  # Initialize the direction p.
  dcopy(n,r,1,p,1)

  # Initialize rho and the norms of r and t.
  rho = ddot(n,r,1,r,1)
  rnorm0 = sqrt(rho)

  # Exit if g = 0.
  iters = 0
  if rnorm0 == zero
    iters = 0
    info = 1
    return info, iters
  end

  for iters=1:itermax

    # Note:
    # Q(w) = 0.5*w'Bw + h'w, where B=L^{-1}AL^{-T}, h=L^{-1}g.
    # Then p'Bp = p'L^{-1}AL^{-T}p = p'L^{-1}Az = p'q.
    # alpha = r'r / p'Bp.

    dcopy(n,p,1,z,1)
    dtsol(n, L, z)

    # Compute q by solving L*q = A*z and save L*q for
    # use in updating the residual t.
    dssyax(n, A, z, q)
    dcopy(n,q,1,z,1)
    dnsol(n, L, q)

    # Compute alpha and determine sigma such that the trust region
    # constraint || w + sigma*p || = delta is satisfied.
    ptq = ddot(n,p,1,q,1)
    if ptq > zero
      alpha = rho/ptq
    else
      alpha = zero
    end
    sigma = dtrqsol(n,w,p,delta)

    # Exit if there is negative curvature or if the
    # iterates exit the trust region.

    if (ptq <= zero) || (alpha >= sigma)
      daxpy(n,sigma,p,1,w,1)
      if ptq <= zero
        info = 3
      else
        info = 4
      end

      return info, iters
    end

    # Update w and the residuals r and t.
    # Note that t = L*r.

    daxpy(n,alpha,p,1,w,1)
    daxpy(n,-alpha,q,1,r,1)
    daxpy(n,-alpha,z,1,t,1)

    # Exit if the residual convergence test is satisfied.

    rtr = ddot(n,r,1,r,1)
    rnorm = sqrt(rtr)
    tnorm = sqrt(ddot(n,t,1,t,1))

    if tnorm <= tol
      info = 1
      return info, iters
    end

    if rnorm <= stol
      info = 2
      return info, iters
    end

    # Compute p = r + beta*p and update rho.
    beta = rtr/rho
    dscal(n,beta,p,1)
    daxpy(n,one,r,1,p,1)
    rho = rtr
  end

  iters = itermax
  info = 5
  return info, iters
end
