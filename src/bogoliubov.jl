"""
```julia
create_bogoliubov_MPO(U::Matrix,V::Matrix,m::Int,sites::Vector{Index{Int64}}; use_gpu::Bool=false, mode="creation")
```
Creates the Bogoliubov operator d†ₘ as MPO like in the reference: https://doi.org/10.1103/PhysRevB.101.165135.

This form is slightly different from the reference, coming from a different Jordan-Wigner convention (see OneNote)

This function can also be used for the Wannier-orbital MPOs or different SU(N) transformations of U and V.

Note:  - as long as U and V are real matrices, `U=conj(U)` and `V=conj(V)`

# Returns
    • MPO(tensor_train)::MPO - The Bogoliubov operator as MPO

# Keyword arguments
    • U::Matrix: Matrix from generalized bogoliubov transformation.
    • V::Matrix: Matrix from generalized bogoliubov transformation.
    • m::Int: Index of the Bogoliubov operator d†ₘ.
    • sites::Vector{Index{Int64}}: The sites object in iTensors format WITHOUT QN.

# Optional keyword arguments
    • use_gpu::Bool=false: If set to true, stores the MPO as CuArray
    • mode::String="creation":  
        - "creation" : creates a creation operator of the given type
        - "annihilation" : creates an annihilation operator of the given type

TODO: Add support for QN sites
...
"""
function create_bogoliubov_MPO(U::Matrix,V::Matrix,m::Int,sites::Vector{Index{Int64}}; use_gpu::Bool=false, mode="creation")
    L = length(sites)

    @assert hastags(sites[1], "Fermion") || hastags(sites[1], "S=1/2") "Only Fermion or S=1/2 sitetypes are possible for MPO-MPS method."

    # for S=1/2 sites we use JW. 
    # Here my convention has σ_minus for V and σ_plus for U.
    # For the fermion sitetype however, σ_minus becomes Cdag which is in matrix form the same as σ_plus. Vice versa for σ_plus and C.
    σ_plus = hastags(sites[1], "Fermion") ? sparse([0 1; 0 0]) : sparse([0 0; 1 0]) # C
    σ_minus = hastags(sites[1], "Fermion") ? sparse([0 0; 1 0]) : sparse([0 1; 0 0]) # Cdag
    σ_z = hastags(sites[1], "Fermion") ? I(2) : sparse([1 0; 0 -1])

	links = map(0:L) do jj
		Index(2, tags = "Link, l=$jj")
	end

    if mode=="annihilation"  #swap U and V
        U_copy = deepcopy(U)
        U = deepcopy(V)
        V = U_copy
    end

    type = isreal(V) ? Float64 : ComplexF64 # only store complex coef if needed

    tensor_train = map(1:L) do j
        A11 = I(2)
		A12 = zeros(2,2)
		A21 = V[j,m] * σ_minus + U[j,m] * σ_plus
        A22 = σ_z
		M = Array{type, 4}(undef, 2,2,2,2)
		M[1,:,:,1] = A11
		M[1,:,:,2] = A12
		M[2,:,:,1] = A21
		M[2,:,:,2] = A22

        Mit = ITensor(M, [links[j], sites[j], prime(sites[j]),  links[j+1]])

        if j==1
            Mit *= onehot(links[1] => 2)
        elseif j==L
            Mit *= onehot(links[end] => 1)
        end
		
		return Mit
	end

    if use_gpu && CUDA.has_cuda() # double check if machine has CUDA available
        return ITensors.adapt(CuArray,MPO(tensor_train))
    else
        return MPO(tensor_train)
    end
end

"""
```julia
getWannier_W_matrix(V,U; kwargs...)
```
Computes the matrix W that diagonalizes the position operator (Reference: https://doi.org/10.1103/PhysRevB.101.165135 page 4)

# Returns
    • W::Matrix: The Matrix that diagonalizes the position operator

# Keyword arguments
    • U::Matrix: Matrix from generalized bogoliubov transformation.
    • V::Matrix: Matrix from generalized bogoliubov transformation.
...
"""
function getWannier_W_matrix(V,U)
    iscomplexmatrix(T) = eltype(T) <: Complex

    V_conj = iscomplexmatrix(V) ? conj(V) : V
    U_conj = iscomplexmatrix(U) ? conj(U) : U

	L = size(V)[1]

    type = isreal(V) ? Float64 : ComplexF64 # only store complex coef if needed

    X = zeros(type, L,L)

    for m in 1:L
        for n in 1:L
            if(m<n)
                for j in 1:L
                    X[m,n] += j*(V[j,n]*V_conj[j,m] + U[j,n]*U_conj[j,m])
                    X[n,m] += j*(V[j,m]*V_conj[j,n] + U[j,m]*U_conj[j,n])
                end
            elseif m==n
                for j in 1:L
                    X[n,n] += j*(V[j,n]*V_conj[j,n] + U[j,n]*U_conj[j,n])
                end
            end
        end
    end 

    # now diagonalize X matrix
    eig = eigen(X)
    W = eig.vectors
    
    @assert W'*X*W ≈ Diagonal(diag(W'*X*W)) "Position operator matrix X was not diagonalized correctly!"

    return W
end