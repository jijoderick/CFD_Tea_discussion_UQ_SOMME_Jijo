"""
Lid-driven cavity flow — DOLFINx version
Solves the unsteady incompressible Navier-Stokes equations on the unit square
using a mixed Taylor-Hood (P2/P1) element and a backward-Euler time discretisation.

Run:
    python3 navier_stokes.py
"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from dolfinx import mesh, fem, io
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from basix.ufl import element, mixed_element
from ufl import (
    TestFunctions,
    split, inner, dot, grad, nabla_grad, div, dx,
)
from petsc4py import PETSc

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
msh = mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)

# ---------------------------------------------------------------------------
# Mixed function space: Taylor-Hood P2/P1
# ---------------------------------------------------------------------------
P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = element("Lagrange", msh.basix_cell(), 1)
TH = mixed_element([P2, P1])
W  = functionspace(msh, TH)

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
# Locate facets on each boundary
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)

# Lid: y == 1  (moving wall)
lid_facets   = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
# Walls: all other boundaries
wall_facets  = mesh.locate_entities_boundary(
    msh, fdim,
    lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)
)

W0 = W.sub(0)   # velocity sub-space
V, V_to_W = W0.collapse()

# Lid velocity: u = (1, 0)
lid_dofs  = locate_dofs_topological((W0, V), fdim, lid_facets)
u_lid     = Function(V)
u_lid.x.array[:] = 0.0
u_lid.x.array[V.dofmap.index_map.size_local * V.dofmap.index_map_bs::] = 0.0
# Set x-component to 1, y-component stays 0
u_lid.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
bc_lid    = dirichletbc(u_lid, lid_dofs, W0)

# No-slip on remaining walls
wall_dofs = locate_dofs_topological((W0, V), fdim, wall_facets)
u_wall    = Function(V)
u_wall.x.array[:] = 0.0
bc_wall   = dirichletbc(u_wall, wall_dofs, W0)

bcs = [bc_lid, bc_wall]

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
w      = Function(W)          # current solution
w_prev = Function(W)          # solution at previous time step
u, p       = split(w)
u_prev, _  = split(w_prev)
v, q       = TestFunctions(W)

# ---------------------------------------------------------------------------
# Physical & time parameters
# ---------------------------------------------------------------------------
nu  = 0.01
rho = 1.0
dt  = 0.1
T   = 2.0

dt_c  = Constant(msh, PETSc.ScalarType(dt))
nu_c  = Constant(msh, PETSc.ScalarType(nu))
rho_c = Constant(msh, PETSc.ScalarType(rho))

# ---------------------------------------------------------------------------
# Variational formulation (backward Euler + convection linearised about u_prev)
# ---------------------------------------------------------------------------
F = (
    rho_c * inner((u - u_prev) / dt_c, v) * dx
    + rho_c * inner(dot(u_prev, nabla_grad(u)), v) * dx
    + nu_c  * inner(grad(u), grad(v)) * dx
    - p * div(v) * dx
    + div(u) * q * dx
)

# ---------------------------------------------------------------------------
# Solver  (v0.9+ API: NonlinearProblem wraps PETSc SNES directly)
# ---------------------------------------------------------------------------
problem = NonlinearProblem(
    F, w,
    petsc_options_prefix="ns_",
    bcs=bcs,
    petsc_options={
        # Newton / SNES (keys are WITHOUT the prefix — it is prepended automatically)
        "snes_type":                    "newtonls",
        "snes_atol":                    1e-12,
        "snes_rtol":                    1e-14,
        "snes_max_it":                  25,
        # Linear sub-solver: MUMPS direct
        "ksp_type":                     "preonly",
        "pc_type":                      "lu",
        "pc_factor_mat_solver_type":    "mumps",
    },
)

# ---------------------------------------------------------------------------
# Output spaces (degree 1 — required by XDMFFile)
# ---------------------------------------------------------------------------
V1_vec = functionspace(msh, element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,)))
V1_sca = functionspace(msh, element("Lagrange", msh.basix_cell(), 1))
u_out = Function(V1_vec, name="velocity")
p_out = Function(V1_sca, name="pressure")

with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as uf, \
     XDMFFile(MPI.COMM_WORLD, "pressure.xdmf", "w") as pf:

    uf.write_mesh(msh)
    pf.write_mesh(msh)

    t = 0.0
    step = 0

    while t < T - 1e-10:
        w_prev.x.array[:] = w.x.array

        problem.solve()

        t    += dt
        step += 1
        print(f"t = {t:.3f}")

        # Extract sub-functions for output
        u_out.interpolate(w.sub(0).collapse())
        p_out.interpolate(w.sub(1).collapse())
        uf.write_function(u_out, t)
        pf.write_function(p_out, t)

# ---------------------------------------------------------------------------
# Quick matplotlib plot of the final velocity magnitude
# ---------------------------------------------------------------------------
import dolfinx.plot as dplt

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    topology, cell_types, geometry = dplt.vtk_mesh(W.sub(0).collapse()[0])
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    u_mag = np.sqrt(np.sum(u_out.x.array.reshape(-1, msh.geometry.dim)**2, axis=1))
    grid["velocity_magnitude"] = u_mag
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, scalars="velocity_magnitude", cmap="viridis")
    plotter.view_xy()
    plotter.screenshot("velocity_magnitude.png")
    print("Saved velocity_magnitude.png")
except Exception:
    # Fallback: plain matplotlib scatter of dof coords coloured by |u|
    coords = W.sub(0).collapse()[0].tabulate_dof_coordinates()
    u_vals = u_out.x.array.reshape(-1, msh.geometry.dim)
    u_mag  = np.linalg.norm(u_vals, axis=1)
    # one point per vector dof — keep only unique xy positions
    xy = coords[:len(u_mag), :2]
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=u_mag, cmap="viridis", s=1)
    plt.colorbar(sc, ax=ax, label="|u|")
    ax.set_title(f"Velocity magnitude at t = {t:.2f}")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("velocity_magnitude.png", dpi=150)
    print("Saved velocity_magnitude.png")