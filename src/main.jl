mutable struct ψ_with_truncerr
    ψ::MPS
    ε_trunc::Float64 # accumulated truncation error of the whole MPO-MPS process
end

"""
```julia
getMpoTrain(U::Matrix,V::Matrix,sites::Vector{Index{Int64}}; type::String="wannier", fillingOrdering::Union{Vector{Int64}, Nothing}=nothing, fillingMode::String="lmr", mode::String="creation", use_gpu::Bool=false)
```

Constructs the MPOs of the MPO-MPS method from the reference: https://doi.org/10.1103/PhysRevB.101.165135 
and stores it as an array with the correct ordering.

# Returns
    • mpoTrain::Array{MPO} - Array that contains the MPOs of the MPO-MPS method in the correct ordering.

# Keyword arguments
    • U::Matrix: Matrix from the bogoliubov transformation.
    • V::Matrix: Matrix from the bogoliubov transformation.
    • sites::Vector{Index{Int64}}: The sites object in ITensors format.

# Optional keyword arguments
    • type::String="bogoliubov":    
        - "bogoliubov" : uses original bogoliubov operators
        - "wannier" : uses maximally localized wannier orbital operators
    • fillingMode::String="ltr":    
        - "ltr" : apply the operators with the left to right scheme
        - "lmr" : apply the operators with the left meet right scheme
    • fillingOrdering::Vector{Int64}=nothing: Specifies a custom order of the bogoliubov operators. If this keyword argument is set the function will ignore the fillingType and fillingMode argument
        (This is important for odd parity cases where the k=0 and k=π mode needs special treatment)
    • mode::String="creation":  
        - "creation" : creates a MPOTrain of creation operators
        - "annihilation" : creates an MPOTrain of annihilation operators
    • use_gpu::Bool=false: If set to true, stores the MPOs as CuArray
"""
function getMpoTrain(
    V::Matrix,
    U::Matrix,
    sites::Vector{Index{Int64}};
    type::String="wannier", 
    fillingOrdering::Union{Vector{Int64}, Nothing}=nothing, 
    fillingMode::String="lmr",
    mode::String="creation",
    use_gpu::Bool=false)

    L = length(sites)

    # wannier type MPOs have different matrix elements than the original bogoliubov
    if type=="wannier"
        W = getWannier_W_matrix(V,U)

        VW_matrix = V*W
        UW_matrix = U*W

        V = VW_matrix
        U = UW_matrix
    end

    if fillingMode == "ltr"
        if isnothing(fillingOrdering)
            fillingOrdering = 1:L
        end
    elseif fillingMode =="lmr"
        allowedSites = isnothing(fillingOrdering) ? (1:L) : fillingOrdering
        fillingOrdering = []

        # now bring to right order for lmr
        for i in 1:div(length(allowedSites) + 1, 2) # divide allowedSites at the half of the array
            push!(fillingOrdering, allowedSites[i]) # add element from first part
            if i != length(allowedSites) - i + 1 # then add element from 2nd part
                push!(fillingOrdering, allowedSites[length(allowedSites) - i + 1])
            end
        end
    end

    mpoTrain = Array{MPO}(fill(MPO(sites),length(fillingOrdering))) 

    for (i,m) in enumerate(fillingOrdering)
        mpoTrain[i] = create_bogoliubov_MPO(U,V,m,sites; use_gpu = use_gpu, mode=mode)
    end

    return mpoTrain
end

"""
```julia
build_groundstate(H_bdg::Matrix;
        maxdim::Int=100,
        cutoff::Float64=1E-13,
        truncation_error::Union{Val{:sum_squared},Val{:accumulated}}=Val(:accumulated),
        groundstate_parity::Union{Val{:even},Val{:odd}}=Val(:even),
        use_gpu::Bool=false, 
        use_fermionic_sitetype::Bool = true,
        fillingOrdering::Union{Vector{Int64},Nothing}=nothing,
        fillingMode::String="lmr",
        type::String="wannier")
```

Creates the ground state for a given fermionic quadratic Hamiltonian matrix in BdG form using the MPO-MPS method (see: https://doi.org/10.1103/PhysRevB.101.165135) which uses parton construction.

# Returns
    • ψ::MPS: The ground state for a given fermionic quadratic Hamiltonian matrix in BdG form as MPS
    • ε_trunc::Float64: The truncation error of the whole MPO-MPS process.

# Keyword arguments
    • H_bdg::Matrix: A fermionic quadratic Hamiltonian as BdG matrix. 

# Optional keyword arguments
    • type::String="wannier":    
        - "bogoliubov" : uses original bogoliubov operators
        - "wannier" : uses maximally localized wannier orbital operators
    • fillingMode::String="lmr":    
        - "ltr" : apply the operators from left to right
        - "lmr" : apply the operators with the left meet right scheme
    • fillingOrdering::Union{Vector{Int64},Nothing}=nothing Specifies the order of the bogoliubov operators to fill the vacuum. 
        (This is important for odd parity cases where the k=0 and k=π mode needs special treatment)
    • maxdim::Int=100: Maximal bond dimension kept by truncation
    • cutoff::Float64=1e-13: Truncation cutoff
    • truncation_error::Union{Val{:sum_squared},Val{:accumulated}}=Val(:accumulated): 
        • :sum_squared - truncation error per step is sum of squared discarded singular values (normalized by the sum of all squared singular values - ITensors default)
        • :accumulated - truncation error is accumulated over all steps as F_acc = ∏ (1 - ε_j) (see: 10.1103/PhysRevB.104.L020409)
        
    • groundstate_parity::Union{Val{:even},Val{:odd}}=Val(:even): Set the ground state parity to be in the correct parity sector
    • use_gpu::Bool=false: If set to true, the MPO-MPS contraction step is moved to GPU 
... 
"""
function build_groundstate(H_bdg::Matrix;
        maxdim::Int=100,
        cutoff::Float64=1E-13,
        truncation_error::Union{Val{:sum_squared},Val{:accumulated}}=Val(:accumulated),
        groundstate_parity::Union{Val{:even},Val{:odd}}=Val(:even),
        use_gpu::Bool=false, 
        use_fermionic_sitetype::Bool = true, # TODO: maybe remove S=1/2 and JW string completely
        fillingOrdering::Union{Vector{Int64},Nothing}=nothing,
        fillingMode::String="lmr",
        type::String="wannier")

    N = size(H_bdg,1) ÷ 2; # number of fermionic modes
    sites_interleaved = use_fermionic_sitetype ? siteinds("Fermion", N) : siteinds("S=1/2", N)

    # Bogoliubov transformation through Eigendecomposition of BdG Hamiltonian
    _, M = eigen(H_bdg)
    V = M[1:N,1:N]
    U = M[N+1:end,1:N]

    # create fermionic vacuum
    vac = begin
        if groundstate_parity == Val(:even) # even parity
            productMPS(sites_interleaved, fill("0", N)) 
        elseif groundstate_parity == Val(:odd) # odd parity TODO: check again if this gives correct results
            productMPS(sites_interleaved, [n!=N ? "0" : "1" for n=1:N]) # all up but last dn
        end
    end

    # move MPS to GPU
    if use_gpu
        vac = ITensors.adapt(CuArray,vac)
    end
    
    # get MPOs in the correct ordering as array
    mpoTrain = getMpoTrain(V,U,sites_interleaved; type=type,fillingMode=fillingMode,fillingOrdering=fillingOrdering,use_gpu=use_gpu)

    # initialize object
    ε_trunc_start = (truncation_error == Val(:accumulated)) ? 1 : 0
    ψ_trial = ψ_with_truncerr(vac, ε_trunc_start)

    # for mpo in mpoTrain
    for mpo in mpoTrain
        apply!(mpo, ψ_trial; maxdim=maxdim, cutoff=cutoff, normalize=true, truncation_error=truncation_error)
    end

    # after apply! ψ_trial.ε_trunc contains F_acc
    if truncation_error == Val(:accumulated)
        ψ_trial.ε_trunc = 1 - ψ_trial.ε_trunc #actual accumulated truncation error
    end
    
    # move back to cpu after contraction
    if use_gpu
        ψ_trial.ψ = ITensors.adapt(Array,ψ_trial.ψ)
    end

    return ψ_trial.ψ, ψ_trial.ε_trunc
end

"""
```julia
build_groundstate(H_bdg::Matrix, sites::MPO;
        maxdim::Int=100,
        cutoff::Float64=1E-13,
        truncation_error::Union{Val{:sum_squared},Val{:accumulated}}=Val(:accumulated),
        groundstate_parity::Union{Val{:even},Val{:odd}}=Val(:even),
        use_gpu::Bool=false, 
        use_fermionic_sitetype::Bool = true,
        fillingOrdering::Union{Vector{Int64},Nothing}=nothing,
        fillingMode::String="lmr",
        type::String="wannier")
```

Creates the ground state for a given fermionic quadratic Hamiltonian matrix in BdG form using the MPO-MPS method (see: https://doi.org/10.1103/PhysRevB.101.165135) which uses parton construction.
Also uses the Gutzwiller projection to enforce single-occupancy constraints on the parton MPS and project the resulting MPS on the physical Hilbert space.

# Returns
    • ψ::MPS: The ground state for a given fermionic quadratic Hamiltonian matrix in BdG form as MPS projected on the physical sites
    • ε_trunc::Float64: The truncation error of the whole MPO-MPS process.

# Keyword arguments
    • H_bdg::Matrix: A fermionic quadratic Hamiltonian as BdG matrix. 
    • sites::MPO: The sites for the MPS AFTER the Gutzwiller projection.

# Optional keyword arguments
    • type::String="wannier":    
        - "bogoliubov" : uses original bogoliubov operators
        - "wannier" : uses maximally localized wannier orbital operators
    • fillingMode::String="lmr":    
        - "ltr" : apply the operators from left to right
        - "lmr" : apply the operators with the left meet right scheme
    • fillingOrdering::Union{Vector{Int64},Nothing}=nothing Specifies the order of the bogoliubov operators to fill the vacuum. 
        (This is important for odd parity cases where the k=0 and k=π mode needs special treatment)
    • maxdim::Int=100: Maximal bond dimension kept by truncation
    • cutoff::Float64=1e-13: Truncation cutoff
    • truncation_error::Union{Val{:sum_squared},Val{:accumulated}}=Val(:accumulated): 
        • :sum_squared - truncation error per step is sum of squared discarded singular values (normalized by the sum of all squared singular values - ITensors default)
        • :accumulated - truncation error is accumulated over all steps as F_acc = ∏ (1 - ε_j) (see: 10.1103/PhysRevB.104.L020409)
        
    • groundstate_parity::Union{Val{:even},Val{:odd}}=Val(:even): Set the ground state parity to be in the correct parity sector
    • use_gpu::Bool=false: If set to true, the MPO-MPS contraction step is moved to GPU 
... 
"""
function build_groundstate(H_bdg::Matrix, sites::Vector{Index{Int64}};
        maxdim::Int=100,
        cutoff::Float64=1E-13,
        truncation_error::Union{Val{:sum_squared},Val{:accumulated}}=Val(:accumulated),
        groundstate_parity::Union{Val{:even},Val{:odd}}=Val(:even),
        use_gpu::Bool=false, 
        use_fermionic_sitetype::Bool = true, # TODO: maybe remove S=1/2 and JW string completely
        fillingOrdering::Union{Vector{Int64},Nothing}=nothing,
        fillingMode::String="lmr",
        type::String="wannier")

    N = size(H_bdg,1) ÷ 2; # number of fermionic modes
    sites_interleaved = use_fermionic_sitetype ? siteinds("Fermion", N) : siteinds("S=1/2", N)

    # Bogoliubov transformation through Eigendecomposition of BdG Hamiltonian
    _, M = eigen(H_bdg)
    V = M[1:N,1:N]
    U = M[N+1:end,1:N]

    # create fermionic vacuum
    vac = begin
        if groundstate_parity == Val(:even) # even parity
            productMPS(sites_interleaved, fill("0", N)) 
        elseif groundstate_parity == Val(:odd) # odd parity TODO: check again if this gives correct results
            productMPS(sites_interleaved, [n!=N ? "0" : "1" for n=1:N]) # all up but last dn
        end
    end

    # move MPS to GPU
    if use_gpu
        vac = ITensors.adapt(CuArray,vac)
    end
    
    # get MPOs in the correct ordering as array
    mpoTrain = getMpoTrain(V,U,sites_interleaved; type=type,fillingMode=fillingMode,fillingOrdering=fillingOrdering,use_gpu=use_gpu)

    # initialize object
    ε_trunc_start = (truncation_error == Val(:accumulated)) ? 1 : 0
    ψ_trial = ψ_with_truncerr(vac, ε_trunc_start)

    # for mpo in mpoTrain
    for mpo in mpoTrain
        apply!(mpo, ψ_trial; maxdim=maxdim, cutoff=cutoff, normalize=true, truncation_error=truncation_error)
    end

    # after apply! ψ_trial.ε_trunc contains F_acc
    if truncation_error == Val(:accumulated)
        ψ_trial.ε_trunc = 1 - ψ_trial.ε_trunc #actual accumulated truncation error
    end
    
    # move back to cpu after contraction
    if use_gpu
        ψ_trial.ψ = ITensors.adapt(Array,ψ_trial.ψ)
    end

    # Gutzwiller projection to enforce single-occupancy constraints and project on physical sites
    n = dim(sites[1]) # number of partons per physical site
    ψ_proj = gutzwillerProjection(sites, ψ_trial.ψ, n)

    return ψ_proj, ψ_trial.ε_trunc
end