#!/bin/bash
# Reproduce every result, table and figure of SOD-ROM v4 and its supplement ("Smooth orthogonal decomposition bases learn static
# correction from response records"). See SOD-ROM/reproduce_sod_rom_v4.pdf for what each step produces.
# Same stages as run_sod_rom_v3.sh, plus the bent-correction tests of Sec. 8.2 (n14, nonlin stage) and Sec. 8.3 (mb10, moon stage).
#
# Usage:  bash run_sod_rom_v4.sh [stage ...]
#   stages: toolbox ensemble geometry ic mix select nonlin moon cost figs   (default: all, in this order)
#   figs only redraws the figures and the cost table (Table S3) from stored outputs (minutes).
# Jobs inside a stage run two at a time (JOBS=2), except the cost stage, which runs alone with one BLAS thread.
# Requirements: Python 3.11+, numpy, scipy, matplotlib, a LaTeX installation (figures use text.usetex).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; ROM="$HERE/rom"; NL="$HERE/nnmrom"; MB="$HERE/moonbeam"
JOBS=${JOBS:-2}
run2 () {  # run the given commands (strings) JOBS at a time, in directory $1
  local dir=$1; shift; cd "$dir"; printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c '{}'
}
STAGES=${@:-toolbox ensemble geometry ic mix select nonlin moon cost figs}
for st in $STAGES; do
  echo "=== stage $st ($(date +%T))"
  case $st in
  toolbox)    # SOD Toolbox against the code it replaced; about 1 min
    cd "$HERE/../SOD-toolbox/python/tests"; python3 test_equivalence.py > test_equivalence_out.txt ;;
  ensemble)   # Secs. 1, 2.2, 4, 5, 8.1, S1, S5, S9 (transfer, noisy training); about 7.5 h
    run2 "$ROM" "python3 c6_pg_vs_eg.py > c6_out.txt" \
                "for i in 0 1 2 3 4 5; do python3 c18_ensemble.py \$i > c18_out_\$i.txt; done" \
                "for i in 2 3 4 5; do python3 c28_fair_transfer.py \$i > c28_out_\$i.txt; done" \
                "python3 c11_timehist.py > c11_out.txt" \
                "for i in 2 3; do python3 c31_noisy_training.py \$i > c31_out_\$i.txt; done"
    cd "$ROM"; python3 c18_summary.py > c18_summary.txt ;;
  geometry)   # Secs. 5-6 (containment, principal angles, LNMs held); minutes. 20 realizations, c18 case-2 seeds
    run2 "$ROM" "python3 g1_static_containment.py 20 > g1_out.txt" "python3 g2_mode_displacement.py 20 > g2_out.txt" \
                "python3 g3_third_direction.py 20 > g3_out.txt" ;;
  ic)         # Sec. 8.1 and S5.1 (which decay to train on); about 30 min
    run2 "$ROM" "for i in 0 1 2 3 4 5; do python3 c10_equal_energy.py \$i > c10_out_\$i.txt; done" \
                "for i in 0 1 2 3 4 5; do python3 c14_bandlimited.py \$i > c14_out_\$i.txt; python3 c15_matched.py \$i > c15_out_\$i.txt; done" \
                "python3 c12_record_energy.py > c12_out.txt; python3 c13_leakage.py > c13_out.txt; python3 c17_ic_projection.py > c17_out.txt"
    cd "$ROM"; python3 c16_ic_summary.py ;;
  mix)        # Secs. 2.3, 3, 6.1, S2, S3, App. A; about 20 min
    run2 "$ROM" "python3 c3_mac.py > c3_out.txt" "python3 c9_state_planes.py > c9_out.txt" "python3 c4_rank.py > c4_out.txt" \
                "python3 c33_unmapped.py > c33_out.txt" "python3 c35_proof_checks.py > c35_out.txt" ;;
  select)     # Sec. 7 and S4 (recognition rules, selection for model reduction); about 1.5 h
    run2 "$ROM" "python3 c19_selection_rules.py > c19_run.log" "python3 c20_state_selection.py > c20_run.log" \
                "python3 c21_rom_selection.py > c21_run.log" "python3 c24_mode1_diag.py > c24_run.log" \
                "python3 c25_width_diag.py > c25_run.log" "python3 c26_circ_rules.py > c26_run.log" ;;
  nonlin)     # Sec. 8.2 and S6 (COST F3 beam); about 1.5 h. Order matters: n3 -> n3b, n4, n5, n6 -> n9a -> n7, n9-n14
    cd "$NL"; python3 n3_backbone_hb.py 8 > n3_out.txt; python3 n3b_check_periodic.py > n3b_out.txt
    run2 "$NL" "python3 n4_rom_backbones.py > n4_out.txt" "python3 n5_equal_energy.py > n5_out.txt" "python3 n6_timehist.py > n6_out.txt"
    python3 n9a_backbone_ext.py > n9a_out.txt
    run2 "$NL" "python3 n7_ssod_companions.py > n7_run.log" "python3 n9_strong_companions.py 0.3 > n9_run_band.log" \
               "python3 n10_plane_span.py > n10_run.log" "python3 n11_diagnose.py > n11_run.log" \
               "python3 n11b_banded_circ.py > n11b_run.log" "python3 n12_backbone_track.py > n12_run.log" "python3 n13_checks.py > n13_run.log"
    python3 n14_bent_mta.py > n14_out.txt 2> n14_run.log    # bent correction vectors, Sec. 8.2 and Table S1; about 2 min
    ;;
  moon)       # Sec. 8.3 and S7 (two-well beam, S5); about 7.5 h. Order: model, full sweeps, bases, ROM sweeps, comparisons
    cd "$MB"; python3 mb_model.py > mb_model_out.txt; python3 mb2_train.py 1 > mb2_out.txt
    run2 "$MB" "python3 mb1_full.py 0.3 > mb1_A0.3.log" "python3 mb6_amp_sweep.py full > mb6_full.log" "python3 mb6z_cascade.py full > mb6z_full.log"
    for b in SOD MTA POD LNM0 LNMw; do
      run2 "$MB" "python3 mb3_rom_sweep.py 0.3 $b > mb3_A0.3_$b.log" "python3 mb6_amp_sweep.py $b > mb6_$b.log" "python3 mb6z_cascade.py $b > mb6z_$b.log"
    done
    cd "$MB"; python3 mb4_compare.py 0.3 > mb4_out.txt; python3 mb7_amp_compare.py > mb7_out.txt; python3 mb8_cascade_compare.py > mb8_out.txt
    python3 mb9_onemode.py > mb9_out.txt
    python3 mb10_dynamic_correction.py > mb10_out.txt     # angles, one-mode table, DMTS bases (mb10_bases.pkl)
    run2 "$MB" "python3 mb10_dynamic_correction.py sweep DMTS 1 > mb10_r1.log" "python3 mb10_dynamic_correction.py sweep DMTS 2 > mb10_r2.log"
    python3 mb7_amp_compare.py > mb7_out_v4.txt          # Table S2: all bases, DMTS included
    ;;
  cost)       # Sec. 9, Sec. S8 and the S4 cost paragraph of Sec. S6. One job at a time, one BLAS thread, nothing else running
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    cd "$ROM"; for k in 0 1; do python3 c36_cost_matched.py $k > c36_out_$k.txt; done
    for k in 0 1; do python3 k1_basis_cost.py $k > k1_out_$k.txt; done; python3 k2_matched_cost.py > k2_out.txt
    cd "$NL"; python3 n2b_cost_best3.py > n2b_out.txt
    unset OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS ;;
  figs)       # all figures and Table S3 from stored outputs; a few minutes. Fig. 1 is TikZ, compiled with the paper
    cd "$ROM"; for f in mkfig_c_error2 mkfig_c_static mkfig_c_mac mkfig_c_traj mkfig_c_ic; do python3 $f.py; done
    python3 mktab_c_cost.py
    cd "$NL";  for f in mkfig_c_nl mkfig_c_nltraj mkfig_c_track; do python3 $f.py; done
    cd "$MB";  python3 mkfig_mb_amp_all.py; python3 mkfig_mb_cascade.py ;;
  *) echo "unknown stage $st"; exit 1 ;;
  esac
done
echo "=== done ($(date +%T)); figures are in figures/, Table S3 in SOD-ROM/tables/, numerical outputs next to each script"
