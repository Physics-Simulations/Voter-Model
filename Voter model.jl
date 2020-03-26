using Plots
using Statistics
using StatsBase

"""
init_lattice(dims:: Array, qs::Array, p::Array)

Initialise the d-dimensional lattice for the Voter model

Arguments
* `dims`: Array of the dimension of the lattice
* `qs`: Array of opinions
* `p`: Array of initial presence probability for each q
Returns
* `lattice` 
"""
function init_lattice(dims, qs, p="Uniform")
    
    if p == "Uniform"
        
        lattice = zeros(dims)
    
        for i in eachindex(lattice)
                
            lattice[i] = rand(qs)
            
        end
        
    else
        println("Feature not implemented yet")
    end
    
    return lattice
    
end

function unit_tuple(D::Int, i::Int, val::Int)
    
    tmp = zeros(Int, D)
    tmp[i] = val
    
    return Tuple(tmp)
    
end

"""
build_neighbours(dims:: Array)

Initialise the d-dimensional lattice for the Voter model

Arguments
* `dims`: Array of the dimension of the lattice
Returns
* `neighbours_table`: where neighbours_table[i][site] is the i nearest neighbour of the site i 
"""

function build_neighbours(dims)
    
    sites = CartesianIndices(dims)

    neighbours_table = [circshift(sites, unit_tuple(length(dims), 1, i)) for i in -1:2:1]

    for j in 2 : length(dims)
        
        append!(neighbours_table, [circshift(sites, unit_tuple(length(dims), j, i)) for i in -1:2:1])
        
    end
    
    return neighbours_table
    
end


function random_imitation(lattice, neighbours_table, site)
   
    rand_neighbour = rand(1 : ndims(lattice) * 2) #Pick randomly one of the neighbours
    
    lattice[site] = lattice[neighbours_table[rand_neighbour][site]] #Adopt its value
    
    return lattice
    
end

function compute_observables(lattice, neighbours_table, sites)
       
    different_links = 0.

    for site in sites

        site_val = lattice[site]

        for nn in neighbours_table

            neighbour_val = lattice[nn[site]]

            if neighbour_val != site_val

                different_links += 1

            end

        end
        
    end
    
    different_links = different_links / 2 #Each link has been counted twice
    
    return different_links
    
end

function simulation(t, lattice, neighbours_table, sites)
    
    density_t = zeros(t)
    
    MC_t = length(lattice)
    
    for k in 1 : t
        
        for i in 1 : MC_t
        
            site = rand(sites)
        
            lattice = random_imitation(lattice, neighbours_table, site)
            
        end
        
        density_t[k] = compute_observables(lattice, neighbours_table, sites)
        
    end
    
    return lattice, density_t / (length(lattice))
    
end

function complete_sim(lattice, neighbours_table, sites)
    
    density_t = []
    
    MC_t = length(lattice)
    
    density = 1.
    k = 0
    
    while density != 0
        
        density = 0.
        
        k += 1
        
        for i in 1 : MC_t
        
            site = rand(sites)
        
            lattice = random_imitation(lattice, neighbours_table, site)
            
        end
        
        density = compute_observables(lattice, neighbours_table, sites)
        
        density_t[k] = density
        
    end
    
    return lattice, density_t / length(lattice)
    
end

function Voter_model(dims, qs, t)
    
    lattice = init_lattice(dims, qs);
    
    neighbours_table = build_neighbours(dims)
    
    sites = CartesianIndices(dims)
    
    lattice, density = simulation(t, lattice, neighbours_table, sites)
    
    return lattice, density
    
end

function Avg_Voter_model(dims, qs, t, times)
    
    final_density = zeros(t)
    taus = zeros(times)
   
    for k in 1 : times
        
        lattice = init_lattice(dims, qs)
    
        neighbours_table = build_neighbours(dims)

        sites = CartesianIndices(dims)

        lattice, density = simulation(t, lattice, neighbours_table, sites)

        final_density += density
        
        taus[k] = length(density[density .> 0])
        
    end
    
    return final_density ./ times, taus
    
end

function N_study(Ns, qs, t, times)
    
    f_tau = open("tau_N_reg_net.txt", "w")
    
    println(f_tau, "#N\t<tau>")
    
    for N in Ns
        
        println("N: $N")
        
        dims = (N, N)
        
        density_t, taus = @time Avg_Voter_model(dims, qs, t, times);
        
        avg_tau = mean(taus)
        
        f = open("results_reg_net_$N.txt", "w")
        
        println(f,"#rho_t")
        
        for i in 1 : length(density_t)
           
            println(f, density_t[i])
            
        end
        
        close(f)
        
        println(f_tau, N, "\t", avg_tau)
                
    end
    
    close(f_tau)
    
end
    
Ns = [20, 40, 60]

qs = [1, 2]

t = 10^5

times = 1000

N_study(Ns, qs, t, times)
