"""
TROTS Proton Optimizer Comparison
---------------------------------------
Compares SlicerRT's IPOPT C++ solver vs PyRadPlan's scipy L-BFGS-B on the same optimization
problem built from TROTS Proton data.

Run inside Slicer's Python console:
    exec(open(r'path/to/compare_optimizers_SlicerRT.py').read())
"""

import os
import h5py
import numpy as np
import scipy.sparse as sps
import scipy.optimize as sopt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ─── Paths ────────────────────────────────────────────────────────────────────

MAT_FILE = r'path/to/Protons_01.mat'
OUT_FILE = r'./dvh_comparison.png'
TXT_FILE = os.path.splitext(OUT_FILE)[0] + '.txt'

_summary = []

# ─── 1. Load TROTS data ───────────────────────────────────────────────────────

def load_trots(mat_file):
    f = h5py.File(mat_file)

    def rstr(ref):
        return ''.join(chr(int(c)) for c in f[ref][()].flatten())

    entries = []
    for i in range(len(f['problem']['dataID'])):
        entries.append({
            'name':          rstr(f['problem']['Name'][i, 0]),
            'data_id':       int(f[f['problem']['dataID'][i, 0]][()].flatten()[0]),
            'is_constraint': bool(f[f['problem']['IsConstraint'][i, 0]][()].flatten()[0]),
            'minimize':      bool(f[f['problem']['Minimise'][i, 0]][()].flatten()[0]),
            'weight':        float(f[f['problem']['Weight'][i, 0]][()].flatten()[0]),
            'bound':         float(f[f['problem']['Objective'][i, 0]][()].flatten()[0]),
        })

    d_mats = {}
    for idx in range(len(f['data']['matrix']['A'])):
        ref = f['data']['matrix']['A'][idx, 0]
        mat = f[ref]
        if mat.attrs.get('MATLAB_class') == b'double':
            data = mat['data'][()]
            ir   = mat['ir'][()]
            jc   = mat['jc'][()]
            n_vox = int(mat.attrs['MATLAB_sparse'])
            n_bix = jc.size - 1
            col = np.zeros(jc[-1], dtype=np.int64)
            for j in range(n_bix):
                col[jc[j]:jc[j+1]] = j
            D = sps.csr_matrix((data, (ir, col)), shape=(n_vox, n_bix))
        else:
            D = sps.csr_matrix(f[ref][()].T)
        d_mats[idx + 1] = D

    f.close()
    return entries, d_mats


print("Loading TROTS data...")
entries, d_mats = load_trots(MAT_FILE)
n_bixels = d_mats[1].shape[1]
print(f"  {len(entries)} problem entries, {n_bixels} bixels")

# ─── 2. Build optimization problem ───────────────────────────────────────────
# Map TROTS entries → pyRadPlan objective objects.
#   is_constraint=True,  minimize=False  → target coverage  → SquaredUnderdosing
#   is_constraint=True,  minimize=True   → OAR max dose     → SquaredOverdosing
#   is_constraint=False, minimize=True   → all objectives   → SquaredOverdosing
#
# Entries with < MIN_VOXELS rows are scenario samples (9 rows) or mean-dose
# aggregates (1 row) used for robustness — skip them here.

from pyRadPlan.optimization.objectives import SquaredOverdosing, SquaredUnderdosing

MIN_VOXELS       = 100   # skip scenario/mean entries
CONSTRAINT_WEIGHT = 10.0  # soft weight given to hard TROTS constraints

struct_terms = []  # (D, pyrad_obj, weight, label)

for e in entries:
    D = d_mats[e['data_id']]
    if D.shape[0] < MIN_VOXELS:
        continue

    if e['is_constraint']:
        w = CONSTRAINT_WEIGHT
        if not e['minimize']:
            obj = SquaredUnderdosing(priority=1.0, d_min=e['bound'])
            label = f"{e['name']} (≥{e['bound']:.1f} Gy)"
        else:
            obj = SquaredOverdosing(priority=1.0, d_max=e['bound'])
            label = f"{e['name']} (≤{e['bound']:.1f} Gy)"
    else:
        w   = e['weight']
        obj = SquaredOverdosing(priority=1.0, d_max=e['bound'])
        label = f"{e['name']} obj≤{e['bound']:.1f}"

    struct_terms.append((D, obj, w, label))

print(f"  {len(struct_terms)} active terms (after skipping mean/scenario entries)")


def total_f(w_flat):
    total = 0.0
    for D, obj, weight, _ in struct_terms:
        dose = D @ w_flat
        total += weight * obj.compute_objective(dose)
    return float(total)


def total_grad(w_flat):
    grad = np.zeros(n_bixels)
    for D, obj, weight, _ in struct_terms:
        dose  = D @ w_flat
        g_d   = obj.compute_gradient(dose)
        grad += weight * (D.T @ g_d)
    return grad


x0 = np.full(n_bixels, 1.0 / n_bixels)
print(f"  Initial objective: {total_f(x0):.4f}")

# ─── 3. Run our C++ IPOPT optimizer via Slicer PythonQt ──────────────────────
# qSlicerIpoptOptimizer is a QObject wrapped by PythonQt.
# Uses addStructureTerm (Qt-compatible types) since PythonQt can't pass Python
# callables to std::function<> parameters.

print("\nRunning our C++ IPOPT optimizer (qSlicerIpoptOptimizer)...")
try:
    import PythonQt
    our_opt = PythonQt.qSlicerExternalBeamPlanningModuleWidgets.qSlicerIpoptOptimizer()
    our_opt.setMaxIterations(10000)

    our_opt.clearStructureTerms()
    for D, obj, weight, label in struct_terms:
        D_coo = D.tocoo()
        obj_type = type(obj).__name__  # "SquaredOverdosing" or "SquaredUnderdosing"
        if hasattr(obj, 'd_max'):
            bound = float(obj.d_max)
        elif hasattr(obj, 'd_min'):
            bound = float(obj.d_min)
        else:
            bound = 0.0
        our_opt.addStructureTerm(
            D_coo.data.tolist(),
            D_coo.row.astype(int).tolist(),
            D_coo.col.astype(int).tolist(),
            int(D.shape[0]), int(D.shape[1]),
            obj_type, bound, float(weight)
        )

    t0 = time.time()
    success = our_opt.solve(x0.tolist())
    t_ours = time.time() - t0
    w_ours = np.asarray(our_opt.getSolution())
    _line = f"  Done in {t_ours:.1f}s  |  success={success}  |  f={total_f(w_ours):.6f}"
    print(_line)
    _summary.append("Running our C++ IPOPT optimizer (qSlicerIpoptOptimizer)...")
    _summary.append(_line)
    has_ours = success
except Exception as e:
    print(f"  qSlicerIpoptOptimizer unavailable ({e})")
    has_ours = False
    w_ours = None

# ─── 4. Run pyRadPlan scipy solver ───────────────────────────────────────────
# Note: pyRadPlan's IPOPT solver requires ipyopt which is not installed.
# OptimizerSciPy uses the same NonLinearOptimizer interface and L-BFGS-B internally.

from pyRadPlan.optimization.solvers._scipy_solver import OptimizerSciPy

pyrad_solver = OptimizerSciPy()
pyrad_solver.objective   = total_f
pyrad_solver.gradient    = total_grad
pyrad_solver.max_iter    = 3000
pyrad_solver.abs_obj_tol = 1e-10
pyrad_solver.options.update({'ftol': 1e-15, 'gtol': 1e-8})

print("\nRunning pyRadPlan scipy solver (L-BFGS-B)...")
t0 = time.time()
w_pyrad, status_pyrad = pyrad_solver.solve(x0)
t_pyrad = time.time() - t0
_line = f"  Done in {t_pyrad:.1f}s  |  success={status_pyrad.success}  |  f={total_f(w_pyrad):.6f}"
print(_line)
_summary.append("\nRunning pyRadPlan scipy solver (L-BFGS-B)...")
_summary.append(_line)

if w_ours is None:
    w_ours = w_pyrad
    t_ours = t_pyrad

# ─── 5. Compute and plot DVH ──────────────────────────────────────────────────
# Key structures: use the constraint D matrices (per-voxel, unscaled)
DVH_STRUCTURES = {}
for e in entries:
    if e['name'] not in DVH_STRUCTURES and d_mats[e['data_id']].shape[0] >= MIN_VOXELS:
        DVH_STRUCTURES[e['name']] = e['data_id']

n_structs = len(DVH_STRUCTURES)
n_cols = 4
n_rows = (n_structs + 1 + n_cols - 1) // n_cols  # +1 for summary table
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.5))
axes = axes.flatten()
fig.suptitle('DVH Comparison: Our C++ IPOPT vs pyRadPlan scipy\n(TROTS Proton 01)', fontsize=13)

label_ours = 'Our C++ IPOPT' if has_ours else 'pyRadPlan scipy (copy)'

for ax, (name, did) in zip(axes, DVH_STRUCTURES.items()):
    D = d_mats[did]
    d_pr = D @ w_pyrad
    d_sc = D @ w_ours

    vmax = max(d_pr.max(), d_sc.max()) * 1.05
    bins = np.linspace(0, vmax, 300)

    dvh_pr = np.array([np.mean(d_pr >= b) for b in bins]) * 100
    dvh_sc = np.array([np.mean(d_sc >= b) for b in bins]) * 100

    ax.plot(bins, dvh_sc, 'r-',  lw=2, label=label_ours)
    ax.plot(bins, dvh_pr, 'b--', lw=2, label='pyRadPlan scipy')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Dose (Gy)', fontsize=8)
    ax.set_ylabel('Volume (%)', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

# Hide unused subplot
for ax in axes[n_structs:]:
    ax.set_visible(False)

# Summary table in last subplot
ax_sum = axes[n_structs]
ax_sum.set_visible(True)
ax_sum.axis('off')
rows = [
    ['Solver', 'Time (s)', 'Final f'],
    [label_ours,        f'{t_ours:.1f}',  f'{total_f(w_ours):.4f}'],
    ['pyRadPlan scipy', f'{t_pyrad:.1f}', f'{total_f(w_pyrad):.4f}'],
]
tbl = ax_sum.table(cellText=rows[1:], colLabels=rows[0],
                   loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2)
ax_sum.set_title('Summary', fontsize=10)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"\nDVH comparison saved to: {OUT_FILE}")

# Also print per-structure mean dose summary
_summary.append("\nMean dose summary (Gy):")
_summary.append(f"{'Structure':35s}  {'pyRadPlan':>12s}  {label_ours[:12]:>12s}  {'diff':>8s}")
_summary.append("-" * 72)
for name, did in DVH_STRUCTURES.items():
    D = d_mats[did]
    md_pr = float((D @ w_pyrad).mean())
    md_sc = float((D @ w_ours).mean())
    _summary.append(f"{name:35s}  {md_pr:12.3f}  {md_sc:12.3f}  {abs(md_pr-md_sc):8.4f}")

print('\n'.join(_summary))

with open(TXT_FILE, 'w') as f:
    f.write('\n'.join(_summary) + '\n')
print(f"Summary saved to: {TXT_FILE}")
