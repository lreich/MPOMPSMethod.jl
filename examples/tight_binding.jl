using MPOMPSMethod
using ITensors, ITensorMPS, SparseArrays

function Tight_binding_MPO(sites::Vector{<:Index}; 
    t::Real=1.0)

    L = length(sites)

    os = OpSum()

    for j in 1:L-1
        os += -t,"S+",j,"S-",j+1
        os += -t,"S-",j,"S+",j+1
    end

    return MPO(os,sites)
end

function Tight_binding_bdg(L::Int=20; t::Real=1.0) 
    
    T = t .* spdiagm(1 => -1 .* ones(L-1), -1 => -1 .* ones(L-1))
    
	H = zeros(Float64, 2*L, 2*L)
	
	H[1:L,1:L] = T
	H[L+1:end,L+1:end] = -transpose(T)
	H[1:L,L+1:end] = zeros(L,L)
	H[L+1:end,1:L] = -zeros(L,L)
	
	return H
end

#= Create model =#
L = 10
t = 1.0
H_bdg = Tight_binding_bdg(L; t=t)

#= Set truncation parameters =#
D_max = 50
cutoff_max = 1e-10

#= MPO-MPS method =#
ψ_trial, _ = build_groundstate(H_bdg; maxdim=D_max, cutoff=cutoff_max, use_fermionic_sitetype=false, fillingOrdering=collect(1:2:L))
sites = siteinds(ψ_trial)

H_mpo = Tight_binding_MPO(sites; t=t)
E0_trial = real(inner(ψ_trial',H_mpo,ψ_trial)) / length(sites)

#= Compare with DMRG result =#
E0_dmrg, psi0_dmrg = dmrg(H_mpo, randomMPS(sites); nsweeps = 10, maxdim = D_max, cutoff=cutoff_max, outputlevel = 1)
E0_dmrg /= length(sites)

@show E0_trial
@show E0_dmrg
@show (E0_trial - E0_dmrg);