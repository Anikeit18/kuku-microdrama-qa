#!/usr/bin/env bash
# Rebuild the test sets and every raw output from scratch.
set -e
cd "$(dirname "$0")"
rm -rf outputs && mkdir -p outputs   # outputs_archive/ is kept
python3 tests/make_testset.py --out samples --set tuning
python3 tests/make_testset.py --out samples_heldout --set heldout
python3 qa.py samples         --scripts samples         --out outputs/tuning_with_script --faces skin
python3 qa.py samples_heldout --scripts samples_heldout --out outputs/heldout_with_script --faces skin
python3 qa.py samples         --out outputs/tuning_no_script --faces skin
python3 qa.py samples_heldout --out outputs/heldout_no_script --faces skin
for d in tuning_with_script tuning_no_script; do
  python3 tests/evaluate.py --samples samples --reports outputs/$d --label $d
done
for d in heldout_with_script heldout_no_script; do
  python3 tests/evaluate.py --samples samples_heldout --reports outputs/$d --label $d
done
