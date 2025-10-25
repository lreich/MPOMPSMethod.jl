function apply!(
    A::MPO,
    ψ_obj::ψ_with_truncerr;
    cutoff=1e-13,
    maxdim=maxlinkdim(A) * maxlinkdim(ψ_obj.ψ),
    mindim=1,
    normalize=false,
    truncation_error::Union{Val{:sum_squared},Val{:accumulated}}=Val(:sum_squared),
    kwargs...,
  )
    ε_j = 0

    n = length(A)
    n != length(ψ_obj.ψ) &&
    throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(ψ_obj.ψ))) do not match"))
    if n == 1
        ψ_obj.ψ = MPS([A[1] * ψ_obj.ψ[1]])
        return nothing
    end
    mindim = max(mindim, 1)
    requested_maxdim = maxdim
    ψ_out = similar(ψ_obj.ψ)

    any(i -> isempty(i), siteinds(commoninds, A, ψ_obj.ψ)) &&
    error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

    # In case A and ψ have the same link indices
    A = sim(linkinds, A)

    ψ_c = dag(ψ_obj.ψ)
    A_c = dag(A)

    # To not clash with the link indices of A and ψ
    sim!(linkinds, A_c)
    sim!(linkinds, ψ_c)
    sim!(siteinds, commoninds, A_c, ψ_c)

    # A version helpful for making the density matrix
    simA_c = sim(siteinds, uniqueinds, A_c, ψ_c)

    # Store the left environment tensors
    E = Vector{ITensor}(undef, n - 1)

    E[1] = ψ_obj.ψ[1] * A[1] * A_c[1] * ψ_c[1]
    for j in 2:(n - 1)
        E[j] = E[j - 1] * ψ_obj.ψ[j] * A[j] * A_c[j] * ψ_c[j]
    end
    R = ψ_obj.ψ[n] * A[n]
    simR_c = ψ_c[n] * simA_c[n]
    ρ = E[n - 1] * R * simR_c
    l = linkind(ψ_obj.ψ, n - 1)
    ts = isnothing(l) ? "" : tags(l)
    Lis = siteinds(uniqueinds, A, ψ_obj.ψ, n)
    Ris = siteinds(uniqueinds, simA_c, ψ_c, n)
    # F_notrunc = eigen(ρ, Lis, Ris; ishermitian=true, tags=ts, cutoff=0, kwargs...)
    # fullSpec = F_notrunc.spec.eigs
    F = eigen(ρ, Lis, Ris; ishermitian=true, tags=ts, cutoff, maxdim, mindim, kwargs...)
    D, U, Ut = F.D, F.V, F.Vt
    
    # see: https://docs.itensor.org/ITensors/stable/examples/ITensor.html#Factoring-ITensors-(SVD,-QR,-etc.)
    ε_j += F.spec.truncerr


    l_renorm, r_renorm = F.l, F.r
    ψ_out[n] = Ut
    R = R * dag(Ut) * ψ_obj.ψ[n - 1] * A[n - 1]
    simR_c = simR_c * U * ψ_c[n - 1] * simA_c[n - 1]
    for j in reverse(2:(n - 1))
        # Determine smallest maxdim to use
        cip = commoninds(ψ_obj.ψ[j], E[j - 1])
        ciA = commoninds(A[j], E[j - 1])
        prod_dims = dim(cip) * dim(ciA)
        maxdim = min(prod_dims, requested_maxdim)
    
        s = siteinds(uniqueinds, A, ψ_obj.ψ, j)
        s̃ = siteinds(uniqueinds, simA_c, ψ_c, j)
        ρ = E[j - 1] * R * simR_c
        l = linkind(ψ_obj.ψ, j - 1)
        ts = isnothing(l) ? "" : tags(l)
        Lis = IndexSet(s..., l_renorm)
        Ris = IndexSet(s̃..., r_renorm)
        # F_notrunc = eigen(ρ, Lis, Ris; ishermitian=true, tags=ts, cutoff=0, kwargs...)
        # fullSpec = F_notrunc.spec.eigs
        F = eigen(ρ, Lis, Ris; ishermitian=true, tags=ts, cutoff, maxdim, mindim, kwargs...)
        D, U, Ut = F.D, F.V, F.Vt

        # see: https://docs.itensor.org/ITensors/stable/examples/ITensor.html#Factoring-ITensors-(SVD,-QR,-etc.)
        ε_j += F.spec.truncerr

        l_renorm, r_renorm = F.l, F.r
        ψ_out[j] = Ut
        R = R * dag(Ut) * ψ_obj.ψ[j - 1] * A[j - 1]
        simR_c = simR_c * U * ψ_c[j - 1] * simA_c[j - 1]
    end
    if normalize
        R ./= norm(R)
    end
    ψ_out[1] = R
    ITensorMPS.setleftlim!(ψ_out, 0)
    ITensorMPS.setrightlim!(ψ_out, 2)

    F_trunc = begin
        if truncation_error == Val(:sum_squared)
            ε_j
        elseif truncation_error == Val(:accumulated)
            1 - ε_j
        end
    end

    # F_trunc = 1 - ε_j

    ψ_obj.ψ = replaceprime(ψ_out, (1 => 0))

    if truncation_error == Val(:sum_squared)
        ψ_obj.ε_trunc += F_trunc
    elseif truncation_error == Val(:accumulated)
        ψ_obj.ε_trunc *= F_trunc
    end
end