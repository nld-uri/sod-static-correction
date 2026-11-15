# Code, stored results and reproduction manual for: Smooth orthogonal decomposition bases learn static correction from response records

D. Chelidze, Nonlinear Dynamics Laboratory, University of Rhode Island.
Article prepared for the *Journal of Sound and Vibration*. Archive: Zenodo, https://doi.org/10.5281/zenodo.24248585 (reserved; published on submission).
Development repository: https://github.com/nld-uri/sod-static-correction. License: MIT (`LICENSE`). Citation: `CITATION.cff`.

Every number, table and figure of the article is produced from the models alone by the Python scripts in `python/`; there are no measured data.
The stored outputs next to each script are the files the article quotes. They let the figures and Table 3 be redrawn without rerunning the simulations.
`SOD-ROM/reproduce_sod_rom_v3.pdf` maps every section, table, figure and quoted number to its script, random seed, output file and run time.

## Contents

| path | contents |
|---|---|
| `python/rom/` | linear benchmarks S1-S3: accuracy ensemble, geometry of the basis, mixing, rank, mode recognition, complex modes, computation cost; shared modules `systems.py`, `romlib.py`, `romlib2.py`, `sellib.py`, `plotprefs.py` |
| `python/nnmrom/` | nonlinear benchmark S4, the COST F3 cantilever: model (`costbeam.py`), Newmark integration (`nlsim.py`), harmonic-balance backbones, ROM backbones, diagnostics, wall times |
| `python/moonbeam/` | nonlinear benchmark S5, the two-well beam: model, training decay, frequency and amplitude sweeps, cascade, one-mode ROMs |
| `python/resid/rlib.py`, `python/trend/t12_trend.py` | two modules from companion studies, needed only by the toolbox equivalence test |
| `python/run_sod_rom_v3.sh` | driver: reruns everything, or chosen stages, in the right order |
| `SOD-toolbox/python/sodtoolbox/` | the SOD Toolbox version used here (configuration and state SOD, recognition rules, bases and ROMs); the scripts add it to the Python path themselves |
| `SOD-toolbox/python/tests/` | equivalence test of the toolbox against the code it replaced, and its recorded output |
| `figures/` | the figures of the article as vector PDF; `figures/tikz/` holds the TikZ source of Figure 1 |
| `SOD-ROM/` | manuscript sources: `sod_rom_v4.tex` with its supplement `sod_rom_v4_supp.tex` (built together by `build_v4.sh`, which resolves the cross-references), the earlier single-document `sod_rom_v3.tex`, their PDFs, the highlights, the cost table (`tables/tab_c_cost.tex`), and the reproduction manual (`reproduce_sod_rom_v3.tex`, `.pdf`; it follows the v3 section numbering) |
| `refs.bib`, `sodmacros.tex` | bibliography and notation macros of the manuscript |
| `requirements.txt` | pinned Python packages |

319 files, 46 MB: 78 scripts and modules in `python/`, 15 in the toolbox, and 198 stored outputs.

## Environment

- Python 3.11 with NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.9 produced the stored outputs of the simulation stages.
- The cost stage (`c36`, `k1`, `k2`, `n2b`) and every wall time printed in the article were run with Python 3.13.18, NumPy 2.5.3 and SciPy 1.18.1 on a two-core Intel Xeon at 2.1 GHz, one BLAS thread. Evaluation and step counts do not depend on the machine; wall times do.
- `requirements.txt` pins NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2.
- Figures set their text with LaTeX (`text.usetex`), so `pdflatex` must be on the path. Figure 1 is TikZ and is compiled with the manuscript (`standalone` package).

## Running

    cd python
    bash run_sod_rom_v3.sh                  # all stages, about 18 h on two cores
    bash run_sod_rom_v3.sh geometry cost    # chosen stages only
    bash run_sod_rom_v3.sh figs             # redraw the figures and Table 3 from stored outputs (minutes)

Stages, in order: `toolbox`, `ensemble`, `geometry`, `ic`, `mix`, `select`, `nonlin`, `moon`, `cost`, `figs`. The cost stage runs alone with one BLAS thread.
To check a rerun, compare each new output file with the stored one (`diff`). Rerunning `mktab_c_cost.py` without its optional machine-label argument drops the trailing comment line of `tab_c_cost.tex`; the table itself is unchanged.

## Checks run on this archive

On 15 Nov 2026, from a fresh copy of this tree (Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1):

- `c35_proof_checks.py`, `c33_unmapped.py` and `mb9_onemode.py` reproduced their stored outputs byte for byte;
- `mktab_c_cost.py` reproduced Table 3 (apart from the machine-label comment above);
- `SOD-toolbox/python/tests/test_equivalence.py`: 28 of 28 checks passed.

The long stages (ensemble, geometry, recognition, nonlinear and moon sweeps, cost) were not rerun for this archive.

## Provenance

The stored outputs of the ensemble and moon-sweep stages were produced before the SOD code moved into the toolbox. The equivalence test checks 28 toolbox functions against the original code: identical, or within 1e-10, except `gsod`, within 1e-6 near the rank cut. After the move the recognition, mode-shape and nonlinear scripts were rerun and no printed number moved; the geometry and cost scripts were written on the toolbox. The reproduction manual lists which stages were and were not rerun.

The identification study of an earlier draft (identification against ERA, damping from a decay) is not part of this article and its scripts are not included.
