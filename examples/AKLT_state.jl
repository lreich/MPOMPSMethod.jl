using MPOMPSMethod
using ITensors, ITensorMPS, SparseArrays, IterTools

# = Create BLBQ Hamiltonian with AKLT parameters as default =#
function BLBQ_Hamiltonian_MPO(sites::Vector{<:Index};
    J::Real=1.0, 
    K::Real=3/2, 
    periodic::Bool=false)

    function jp(j,i,N) return mod( (j+i) - 1, N) + 1 end

    @assert hastags(sites[1], "S=1") "The sites must be spin-1 sites for the BLBQ model."

    L = length(sites)
    L_bounds = periodic ? L : L - 1
    site_ordering = collect(1:L)
    
    # Bilinear term
    os1 = OpSum()
    # Biquadratic term
    os2 = OpSum()
    for i in 1:L_bounds
        j = site_ordering[i]
        jp1 = site_ordering[jp(i,1,L)]

        # Bilinear term
        os1 += 1/2 * J,"S+",j,"S-",jp1
        os1 += 1/2 * J,"S-",j,"S+",jp1
        os1 += J,"Sz",j,"Sz",jp1

        # Biquadratic term
        os2 += 1/4 * K,"S+",j,"S-",jp1,"S+",j,"S-",jp1
        os2 += 1/4 * K,"S+",j,"S-",jp1,"S-",j,"S+",jp1
        os2 += 1/4 * K,"S-",j,"S+",jp1,"S+",j,"S-",jp1
        os2 += 1/4 * K,"S-",j,"S+",jp1,"S-",j,"S+",jp1

        os2 += 1/2 * K,"S+",j,"S-",jp1,"Sz",j,"Sz",jp1
        os2 += 1/2 * K,"S-",j,"S+",jp1,"Sz",j,"Sz",jp1

        os2 += 1/2 * K,"Sz",j,"Sz",jp1,"S+",j,"S-",jp1
        os2 += 1/2 * K,"Sz",j,"Sz",jp1,"S-",j,"S+",jp1

        os2 += K,"Sz",j,"Sz",jp1,"Sz",j,"Sz",jp1
    end

    os = os1 + os2
    
    return MPO(os, sites)
end;

# = Create BLBQ Hamiltonian in BdG form with AKLT parameters as default =#
function BLBQ_Hamiltonian_bdg(; 
    λ::Real=0.0, 
    Δ::Real=3/2, 
    χ::Real=1.0, 
    J::Real=1.0, 
    K::Real=1/3, 
    N::Int=10, 
    periodic::Bool = false, 
    parity::Int = 1)

    T = spdiagm(0 => λ .* ones(N), 3 => -χ*J .* ones(N-3), -3 => -χ*J .* ones(N-3))
    D = spdiagm(3 => -(J-K)*Δ .* ones(N-3), -3 => (J-K)*Δ .* ones(N-3))

    if periodic
        T[1, N-2] = χ*J * parity
        T[2, N-1] = χ*J * parity
        T[3, N] = χ*J * parity
        T[N-2, 1] = χ*J * parity
        T[N-1, 2] = χ*J * parity
        T[N, 3] = χ*J * parity

        D[1, N-2] = -(J-K)*Δ * parity
        D[2, N-1] = -(J-K)*Δ* parity
        D[3, N] = -(J-K)*Δ * parity
        D[N-2, 1] = (J-K)*Δ * parity
        D[N-1, 2] = (J-K)*Δ * parity
        D[N, 3] = (J-K)*Δ * parity
    end

	H = zeros(Float64,2*N, 2*N)
	
	H[1:N,1:N] = T
	H[N+1:end,N+1:end] = -transpose(T)
	H[1:N,N+1:end] = D
	H[N+1:end,1:N] = -D
    
	return H
end

#= Maps the cartesian coordinate representation after Gutzwiller projection back to the original spin-1 representation =#
function cartesianToSpin1(A::MPS)
    L = length(A)

    B_array= map(1:L) do j
        B = A[j]

        siteIndices = [i for i in inds(B) if hastags(i, "Site")]
        linkIndices = [i for i in inds(B) if hastags(i, "Link")]

        B_mapped = ITensor(ComplexF64,(siteIndices,linkIndices)) # creates empty tensor with same structure as B

        # creates all pairs for the links
        allLinkPointer = map(1:length(linkIndices)) do i
            return map(1:dim(linkIndices[i])) do k
                return linkIndices[i] => k
            end
        end
        allLinksComb = collect(IterTools.product(allLinkPointer...)) # all permutations of linkPointer

        for s in 1:3
            for linkPointer in allLinksComb
                coef = B[siteIndices[1]=>s, linkPointer...]  

                pointerB_Plus = [siteIndices[1]=>1, linkPointer...]
                pointerB_Null = [siteIndices[1]=>2, linkPointer...]
                pointerB_Minus = [siteIndices[1]=>3, linkPointer...]

                if s==1 # tau = x 
                    B_mapped[pointerB_Minus...] += coef/sqrt(2) #  |001>
                    B_mapped[pointerB_Plus...] += -coef/sqrt(2) # -|100>
                    # B_mapped[pointerB_Plus...] += 1im*coef/sqrt(2) #  |001>
                    # B_mapped[pointerB_Minus...] += -1im*coef/sqrt(2) # -|100>
                end
                if s==2 # tau = y 
                    B_mapped[pointerB_Minus...] += 1im*coef/sqrt(2) # |001>
                    B_mapped[pointerB_Plus...] += 1im*coef/sqrt(2) # |100>
                    # B_mapped[pointerB_Plus...] += coef/sqrt(2) # |001>
                    # B_mapped[pointerB_Minus...] += coef/sqrt(2) # |100>
                end
                if s==3 # tau = z 
                    B_mapped[pointerB_Null...] += coef #  |010>
                    # B_mapped[pointerB_Null...] += -1im*coef #  |010>
                end
            end
        end
        return B_mapped
    end

    return MPS(B_array) 
end

#= Create model at the AKLT point =#
L = 10
J = 1.0
K = 1/3
Δ = 3/2
χ = 1.0
λ = 0.0

n = 3 # spin-1 -> 3 partons per site
N = n * L

sites = siteinds("S=1", L)

H_bdg = BLBQ_Hamiltonian_bdg(λ=λ, Δ=Δ, χ=χ, J=J, K=K, N=N, periodic=true, parity=1)

#= Set truncation parameters =#
D_max = 50
cutoff_max = 1e-10

#= MPO-MPS method =#
ψ_trial, _ = build_groundstate(H_bdg; maxdim=D_max, cutoff=cutoff_max, use_fermionic_sitetype=false)

# Gutzwiller projection
ψ_trial = gutzwillerProjection(sites, ψ_trial, n);

# Map back to spin-1 representation
ψ_trial = cartesianToSpin1(ψ_trial);

H_mpo = BLBQ_Hamiltonian_MPO(sites; J=J, K=K, periodic=true)
E0_trial = real(inner(ψ_trial',H_mpo,ψ_trial)) / length(sites)

#= Compare with DMRG result =#
E0_dmrg, psi0_dmrg = dmrg(H_mpo, randomMPS(sites); nsweeps = 10, maxdim = D_max, cutoff=cutoff_max, outputlevel = 1)
E0_dmrg /= length(sites)

E_exact = -2/3
@show E0_trial
@show E0_dmrg
@show E_exact
@show (E0_trial - E0_dmrg);