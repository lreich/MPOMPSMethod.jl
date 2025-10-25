"""
```julia
gutzwillerProjection_MPO(sites_interleaved::Vector{Index{Int64}}, n::Int)
```
Returns an MPO P_G acting on the interleaved parton chain that enforces the local single-occupancy constraint per physical site.

# Assumptions
- Partons belonging to a physical site are adjacent in `sites_interleaved`.
- `n` is the number of partons per physical site (2 for S=1/2 or 3 for S=1).

# Effect
- For each block of n adjacent parton sites, P_G projects onto allowed configurations:
  - n=2: {|10>, |01>} kept, {|00>, |11>} annihilated
  - n=3: {|100>, |010>, |001>} kept, others annihilated

# Example
- Apply with `apply(P, psi; ...)`; you can then map to physical sites with `gutzwillerProjection` if needed.
"""
function gutzwillerProjection_MPO(sites_interleaved::Vector{Index{Int64}}, n::Int)
    N = length(sites_interleaved)
    @assert N % n == 0 "Length of interleaved sites must be a multiple of partons per site"
    @assert dim(sites_interleaved[1]) == 2 "Parton sites must have dimension 2 (occupied / unoccupied)"
    L = N ÷ n  # number of physical sites
    
    # Create link indices - we need link dimension > 1 to handle the constraints
    links = [Index(n+1, tags="Link,l=$j") for j in 0:N]
    
    tensor_train = map(1:N) do j
        s = sites_interleaved[j]
        spr = prime(s)
        
        # Determine position within interleaved lattice block
        pos_in_block = mod(j-1, n) + 1
        
        if n == 2
            # For 2-parton case: we want to allow |10> and |01>, forbid |00> and |11>
            # Use link states: 1=start, 2=after first occupied, 3=after two occupied (illegal)
            
            M = zeros(n+1, 2, 2, n+1)  # (link_left, s, s', link_right)
            
            if pos_in_block == 1  # First parton of the pair
                # From start state (1):
                M[1, 1, 1, 1] = 1.0  # |0> -> stay in start state
                M[1, 2, 2, 2] = 1.0  # |1> -> go to "one occupied" state
                
                # From "one occupied" state (2): (shouldn't happen for first parton)
                M[2, 1, 1, 2] = 1.0  # |0> -> stay in "one occupied" 
                M[2, 2, 2, 3] = 1.0  # |1> -> go to "two occupied" (illegal)
                
                # From "two occupied" state (3): everything illegal
                M[3, 1, 1, 3] = 0.0  # project out
                M[3, 2, 2, 3] = 0.0  # project out
                
            else  # pos_in_block == 2, Second parton of the pair
                # From start state (1): second parton must be occupied
                M[1, 1, 1, 1] = 0.0  # |0> after |0> = |00> -> project out
                M[1, 2, 2, 1] = 1.0  # |1> after |0> = |01> -> allowed, back to start
                
                # From "one occupied" state (2): second parton must be empty  
                M[2, 1, 1, 1] = 1.0  # |0> after |1> = |10> -> allowed, back to start
                M[2, 2, 2, 1] = 0.0  # |1> after |1> = |11> -> project out
                
                # From "two occupied" state (3): everything already illegal
                M[3, 1, 1, 1] = 0.0  
                M[3, 2, 2, 1] = 0.0  
            end
            
        elseif n == 3
            # For 3-parton case: similar logic but more states
            # Link states: 1=start, 2=one occupied, 3=two occupied (illegal), 4=illegal continuation
            M = zeros(n+1, 2, 2, n+1)
            
            if pos_in_block == 1  # First parton
                M[1, 1, 1, 1] = 1.0  # |0> -> start
                M[1, 2, 2, 2] = 1.0  # |1> -> one occupied
                
            elseif pos_in_block == 2  # Second parton  
                M[1, 1, 1, 1] = 1.0  # |0> after |0> -> start
                M[1, 2, 2, 2] = 1.0  # |1> after |0> -> one occupied
                M[2, 1, 1, 2] = 1.0  # |0> after |1> -> one occupied  
                M[2, 2, 2, 3] = 1.0  # |1> after |1> -> two occupied (will be projected)
                
            else  # pos_in_block == 3, Third parton
                M[1, 1, 1, 1] = 0.0  # |000> -> project out
                M[1, 2, 2, 1] = 1.0  # |001> -> allowed
                M[2, 1, 1, 1] = 1.0  # |010> or |100> -> allowed  
                M[2, 2, 2, 1] = 0.0  # |011> or |101> -> project out
                M[3, 1, 1, 1] = 0.0  # already illegal
                M[3, 2, 2, 1] = 0.0  # already illegal
            end
            
        else
            error("Unsupported number of partons per site n=$n. Only n=2 or n=3.")
        end
        
        # Create the ITensor and handle boundary conditions
        tensor = ITensor(M, [links[j], s, spr, links[j+1]])
        
        if j == 1
            tensor *= onehot(links[1] => 1)  # Start in initial state (0 partons)
        elseif j == N
            # At the end, we should be back in the start state for proper termination
            tensor *= onehot(links[N+1] => 1)
        end
        
        return tensor
    end
    
    return MPO(tensor_train)
end

"""
```julia
gutzwillerProjection(sites::Vector{Index{Int64}}, psi::MPS, n::Int)
```
Project an interleaved parton MPS with the Gutzwiller constraint and cast it on the
target physical sites.

# Arguments
- `sites`: physical site indices of length `L`, each having dimension `n`.
- `psi`: MPS defined on an interleaved parton chain of length `L * n`.
- `n`: number of partons per physical site (supported values: 2 or 3).

# Returns
- `MPS` on `sites` with the single-occupancy constraint enforced per physical site.

# Notes
- The projection is performed without additional truncation and assumes the parton
    indices in `psi` are ordered in contiguous blocks per physical site.
"""
function gutzwillerProjection(sites::Vector{Index{Int64}}, psi::MPS, n::Int)
    L = length(sites)
    N = length(psi)
    @assert N % n == 0 "Length of interleaved MPS must be a multiple of partons per site"
    @assert L * n == N "Number of physical sites times partons per site must match MPS length"
    @assert all(dim(s) == n for s in sites) "Physical site dimension must equal number of partons per site"
    
    sites_interleaved = siteinds(psi)
    P_G = gutzwillerProjection_MPO(sites_interleaved, n)

    # Do not truncate here
    psi_projected = apply(P_G, psi; cutoff=0, normalize=true)

    phys_tensors = Vector{ITensor}(undef, L)

    # Combine each block of n parton tensors into the target physical site index.
    for ell in 1:L
        first = (ell-1) * n + 1
        last = ell * n

        block_tensor = psi_projected[first]
        for j in (first+1):last
            block_tensor *= psi_projected[j]
        end

        block_sites = sites_interleaved[first:last]
        mapper = ITensor(sites[ell], block_sites...)
        if n == 2
            mapper[sites[ell] => 1, block_sites[1] => 2, block_sites[2] => 1] = 1.0
            mapper[sites[ell] => 2, block_sites[1] => 1, block_sites[2] => 2] = 1.0
        elseif n == 3
            mapper[sites[ell] => 1, block_sites[1] => 2, block_sites[2] => 1, block_sites[3] => 1] = 1.0
            mapper[sites[ell] => 2, block_sites[1] => 1, block_sites[2] => 2, block_sites[3] => 1] = 1.0
            mapper[sites[ell] => 3, block_sites[1] => 1, block_sites[2] => 1, block_sites[3] => 2] = 1.0
        else
            error("Unsupported number of partons per site n=$n. Only n=2 or n=3.")
        end

        phys_tensors[ell] = block_tensor * mapper
    end

    psi_final = MPS(phys_tensors)
    # Normalize final MPS
    psi_final /= norm(psi_final)

    return psi_final
end