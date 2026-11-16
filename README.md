# Code, stored results and reproduction manual for: What a free decay teaches a smooth orthogonal decomposition basis

D. Chelidze, Nonlinear Dynamics Laboratory, University of Rhode Island.
Article prepared for the *Journal of Sound and Vibration*. Archive: Zenodo, https://doi.org/10.5281/zenodo.24248585 (published on submission).
Development repository: https://github.com/nld-uri/sod-static-correction. License: MIT (`LICENSE`). Citation: `CITATION.cff`.

Every number, table and figure of the article is produced from the models alone by the Python scripts in `python/`; there are no measured data.
The stored outputs next to each script are the files the article quotes. They let the figures and the cost table (Table S4) be redrawn without rerunning the simulations.
The submitted manuscript is `SOD-ROM/sod_rom_v5.tex` with its supplement `SOD-ROM/sod_rom_v5_supp.tex`. `SOD-ROM/reproduce_sod_rom_v5.pdf` maps every section, table, figure and quoted number of the main text and supplement to its script, random seed, output file and run time; `python/run_sod_rom_v5.sh` reruns everything. Earlier drafts (v3, v4), their manuals and drivers are kept for the record.

## Contents

| path | contents |
|---|---|
| `python/rom/` | linear benchmarks S1-S3: accuracy ensemble, geometry of the basis, mixing, rank, mode recognition, complex modes, computation cost; shared modules `systems.py`, `romlib.py`, `romlib2.py`, `sellib.py`, `plotprefs.py` |
| `python/nnmrom/` | nonlinear benchmark S4, the COST F3 cantilever: model (`costbeam.py`), Newmark integration (`nlsim.py`), harmonic-balance backbones, ROM backbones, diagnostics, wall times |
| `python/moonbeam/` | nonlinear benchmark S5, the two-well beam: model, training decay, frequency and amplitude sweeps, cascade, one-mode ROMs |
| `python/resid/rlib.py`, `python/trend/t12_trend.py` | two modules from companion studies, needed only by the toolbox equivalence test |
| `python/identity/` | the covariance identity, the free-decay angle tests, the force-location scan and the three recognition routes on the dashpot beam (`i1`-`i7`, `i9`); `i8` is an interpolation check the article does not use |
| `python/run_sod_rom_v5.sh` | driver: reruns everything, or chosen stages, in the right order (`run_sod_rom_v4.sh` and `run_sod_rom_v3.sh` are the drivers of the earlier drafts) |
| `SOD-toolbox/python/sodtoolbox/` | the SOD Toolbox version used here (configuration and state SOD, recognition rules, bases and ROMs); the scripts add it to the Python path themselves |
| `SOD-toolbox/python/tests/` | equivalence test of the toolbox against the code it replaced, and its recorded output |
| `figures/` | the figures of the article as vector PDF; `figures/tikz/` holds the TikZ source of Figure 1 |
| `SOD-ROM/` | manuscript sources: the submitted `sod_rom_v5.tex` with its supplement `sod_rom_v5_supp.tex` (built together by `build_v5.sh`, which resolves the cross-references), the earlier drafts `sod_rom_v4.tex` and `sod_rom_v3.tex`, their PDFs, the highlights, the cost table (`tables/tab_c_cost.tex`), and the reproduction manuals (`reproduce_sod_rom_v5.tex`, and those of v4 and v3, with PDFs) |
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
    bash run_sod_rom_v5.sh                  # all stages, about 19.5 h on two cores
    bash run_sod_rom_v5.sh identity cost    # chosen stages only
    bash run_sod_rom_v5.sh figs             # redraw the figures and Table S4 from stored outputs (minutes)

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
