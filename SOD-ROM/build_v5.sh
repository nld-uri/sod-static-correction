#!/bin/bash
# build main and supplement with xr cross-references: $1 = SOD-ROM dir
cd "$1"; rm -f sod_rom_v5*.{aux,bbl,blg,out,log}
run(){ pdflatex -interaction=nonstopmode $1.tex >/dev/null 2>&1 || true; }
run sod_rom_v5; run sod_rom_v5_supp; bibtex sod_rom_v5 >/dev/null; bibtex sod_rom_v5_supp >/dev/null
for i in 1 2 3; do run sod_rom_v5_supp; run sod_rom_v5; done
for d in sod_rom_v5 sod_rom_v5_supp; do echo "== $d undefined:$(grep -c 'undefined' $d.log) multiply:$(grep -ci 'multiply' $d.log) rerun:$(grep -c 'Rerun to' $d.log) errors:$(grep -c '^! ' $d.log) $(pdfinfo $d.pdf | grep Pages)"; done
