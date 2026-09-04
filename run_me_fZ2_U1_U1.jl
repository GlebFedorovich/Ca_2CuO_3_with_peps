using Random
using LinearAlgebra
using PEPSKit
using TensorKit
using TensorKitTensors.HubbardOperators
using OptimKit
using JLD2
using Logging, LoggingExtras
using LinearAlgebra
using MPSKit

BLAS.set_num_threads(1) 

Random.seed!(1234)

mkpath("output")
logname = "output/run_log_temp.log"
file_logger = MinLevelLogger(FileLogger(logname; always_flush = true), Logging.Info)
global_logger(TeeLogger(file_logger, global_logger()))

"""
    hubbard_2x2([T]; t_hor, t_vert, U, V, mu, lattice)

Fermi–Hubbard Hamiltonian on the infinite square lattice with a 2×2 unit cell,
assembled from PEPSKit / TensorKitTensors fermionic (parity-graded) operators:

```math
H = -t_hor  Σ_{⟨ij⟩ₕ} Σ_σ (c†_{iσ} c_{jσ} + h.c.)
    -t_vert Σ_{⟨ij⟩ᵥ} Σ_σ (c†_{iσ} c_{jσ} + h.c.)
    + U Σ_i n_{i↑} n_{i↓}
    + V Σ_{⟨ij⟩}  n_i n_j
    - μ Σ_i n_i
```

* `t_hor`  – hopping across horizontal bonds (columns, `+CartesianIndex(0,1)`)
* `t_perp` – hopping across vertical bonds  (rows,    `+CartesianIndex(1,0)`)
* 't_2'    – next-nearest-neighbour hopping (over 2 sites, `+CartesianIndex(0,2)`)
* `U`      – on-site Hubbard repulsion `n↑ n↓`
* `V_hor`  – horizontal nearest-neighbour density–density interaction `n_i n_j`
* `V_perp` – perpendicular nearest-neighbour density–density interaction `n_i n_j`
* 'V_2'    – next-nearest-neighbour density–density interaction `n_i n_j`
* `mu`     – chemical potential

Returns a `PEPSKit.LocalOperator` ready for CTMRG / PEPS optimization.
"""
    
function hubbard_2x2(
        T::Type{<:Number} = ComplexF64;
        t_hor = 1.0, t_perp = 1.0, t_2 = 0.0, U = 8.0, V_hor = 0.0, V_perp = 0.0, V_2 = 0.0, mu = 0.0,
        lattice::InfiniteSquare = InfiniteSquare(2, 2),
    )
    particle_symmetry = U1Irrep
    spin_symmetry = U1Irrep

    # --- one-site operators --------------------------------------------------
    n_op  = e_num(T, particle_symmetry, spin_symmetry)    # n = n↑ + n↓
    nn_op = ud_num(T, particle_symmetry, spin_symmetry)   # n↑ n↓  (double occupancy)
    pspace = space(n_op, 1)
    unit = id(pspace)

    # --- two-site operators ------------------------------------------------- -
    hop  = e_hopping(T, particle_symmetry, spin_symmetry)  # Σ_σ (c†_{iσ}c_{jσ} + h.c.)
    dens = n_op ⊗ n_op                                     # density–density

    # on-site term, shared evenly over the 4 bonds meeting at each site
    onsite = U * nn_op - mu * n_op
    onsite_bond = (1 // 4) * (onsite ⊗ unit + unit ⊗ onsite)

    h_hor  = (-t_hor)  * hop + V_hor * dens + onsite_bond
    h_perp = (-t_perp) * hop + V_perp * dens + onsite_bond
    h_2    = (-t_2)    * hop + V_2 * dens + onsite_bond

    spaces = fill(pspace, size(lattice))

    terms = []
    for I in vertices(lattice)
        push!(terms, [I, I + CartesianIndex(0, 1)] => h_hor)   # horizontal bond
        push!(terms, [I, I + CartesianIndex(1, 0)] => h_perp)  # vertical bond
        # push!(terms, [I, I + CartesianIndex(0, 2)] => h_2)   # next-nearest-neighbour bond swithed off for now
    end
    return LocalOperator(spaces, terms...)
end

function measure_electron_density(peps, env)
    pspaces = physicalspace(peps)
    return map(vertices(lattice)) do I
        # use the per-site density operator carrying the fused auxiliary charge
        op = n_density.terms[[I]]
        real(expectation_value(peps, LocalOperator(pspaces, [I] => op), env))
    end
end


lattice = InfiniteSquare(2, 2);
H_init = hubbard_2x2(; t_hor = 0.487, t_perp = 0.042, t_2 = 0.0, U = 3.593, V_hor = 1.026, V_perp = 0.841, V_2 = 0.5, mu = 0.0, lattice);

# define the fermionic symmetry and the auxiliary charge matrix for the physical space
fermion = fℤ₂
particle_symmetry = U1Irrep
spin_symmetry = U1Irrep
S = fermion ⊠ particle_symmetry ⊠ spin_symmetry

S_aux_matrix = [S(1,1,1/2) S(1,1,-1/2);
                S(1,1,-1/2) S(1,1,1/2)];
H = MPSKit.add_physical_charge(H_init, S_aux_matrix);

# electron-density operator, charged the same way as the Hamiltonian so it acts on
# the fused physical spaces of the PEPS
n_density_op = e_num(ComplexF64, particle_symmetry, spin_symmetry)
n_density_init = LocalOperator(
    fill(space(n_density_op, 1), size(lattice)),
    ([I] => n_density_op for I in vertices(lattice))...,
);
n_density = MPSKit.add_physical_charge(n_density_init, S_aux_matrix);


# =============================================================================
# Preparing the PEPS ansatz and environment spaces
# =============================================================================

Dbond = 1      # virtual bond dimension scale
χenv  = 32   # CTMRG environment dimension in total

V_peps = Vect[S]((0,0,0) => Dbond, (1,1,1/2) => Dbond, (1,1,-1/2) => Dbond);
V_env  = Vect[S]((0,0,0) => χenv ÷ 4,  (1,1,1/2) => χenv ÷ 4, (1,1,-1/2) => χenv ÷ 4, (0,2,0) => χenv ÷ 4);

physical_spaces = physicalspace(H)
virtual_spaces  = fill(V_peps, size(lattice)...)

peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces);

# =============================================================================
# Define main optimization algorithms and parameters
# =============================================================================

boundary_alg = (;
    tol   = 1.0e-8,
    alg   = :SimultaneousCTMRG,
    trunc = (; alg = :FixedSpaceTruncation),
    verbosity = 3,
)

gradient_alg = (;
    tol        = 1.0e-7,
    maxiter    = 30,
    solver_alg = (; alg = :Arnoldi),
    verbosity = 3,
)
    
# OptimKit L-BFGS optimizer
optimizer_alg = (;
    alg          = :LBFGS,
    tol          = 1.0e-4,
    maxiter      = 10,
    lbfgs_memory = 24,
    ls_maxiter   = 10,
    ls_maxfg     = 10,
    verbosity = 2,
)

reuse_env = true
verbosity = 3

# -----------------------------------------------------------------------------
#  Custom finalize!: after every accepted optimization step, re-shuffle charges
#  in the PEPS environment
# -----------------------------------------------------------------------------

χ_refine = χenv
refine_boundary_alg = (;
    alg           = :SimultaneousCTMRG,
    maxiter       = 20,
    miniter       = 20,        # force exactly 20 sweeps (no early convergence exit)
    tol           = 0.0,
    trunc         = truncrank(χ_refine),
    verbosity     = 3,
)


finalize_logfile = "output/finalize_metrics.txt"

function refine_env_finalize!((peps, env), E, grad, numiter)
    println("refine_env_finalize!: re-contracting PEPS env at $numiter optimization step")
    env_new, _ = leading_boundary(env, peps; refine_boundary_alg...)

    # --- record energy, gradient norm and electron density -----------------
    gradnorm = sqrt(real(dot(grad, grad)))
    densities = measure_electron_density(peps, env_new)
    avg_density = sum(densities) / length(densities)
    site_densities = join(
        ("$(Tuple(I))=>$(round(d; digits = 3))" for (I, d) in zip(vertices(lattice), densities)),
        ", ",
    )

    metric_row(step, energy, gnorm, avg, sites) = string(
        rpad(step, 7), rpad(energy, 12), rpad(gnorm, 12), rpad(avg, 13), sites,
    )

    if !isfile(finalize_logfile)
        open(finalize_logfile, "w") do io
            println(io, metric_row("# step", "energy", "gradnorm", "avg_density", "site_densities"))
        end
    end
    open(finalize_logfile, "a") do io
        println(io, metric_row(
            numiter,
            round(real(E); digits = 4),
            round(gradnorm; digits = 4),
            round(avg_density; digits = 4),
            site_densities,
        ))
    end

    return (peps, env_new), E, grad
end

# converge an initial environment on peps₀
env_warm_up, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; refine_boundary_alg...);
env₀, = leading_boundary(env_warm_up, peps₀; boundary_alg...);

sum(measure_electron_density(peps₀, env₀))  # sanity check: electron density on each site of the unit cell

# =============================================================================
# Start the main fixed-point optimization loop
# =============================================================================

peps, env, E, info = fixedpoint(
    H, peps₀, env₀;
    boundary_alg = (;
        boundary_alg...,          # base CTMRG settings from above
        dynamic_tols = false,      # disable dynamic tolerance scaling for now (default: true)
    ),
    gradient_alg, optimizer_alg, reuse_env, verbosity,
    finalize! = refine_env_finalize!,
)

save("output/peps_final.jld2", Dict("peps" => peps, "env" => env, "E" => E, "info" => info));

