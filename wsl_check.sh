#!/usr/bin/env bash
set -euo pipefail

sudo mkdir -p /mnt/z /mnt/w /mnt/y

for L in Z W Y; do
  m="/mnt/$(echo "$L" | tr '[:upper:]' '[:lower:]')"
  if mountpoint -q "$m" 2>/dev/null; then
    echo "already_$L"
  elif sudo mount -t drvfs "${L}:" "$m" 2>/dev/null; then
    echo "mounted_$L"
  else
    echo "fail_$L"
  fi
done

echo "=== /mnt ==="
ls /mnt/

echo "=== wslpath ==="
wslpath -u 'Z:\MRIdata\cm_monkey_qst_bids' || true
wslpath -u 'D:\imaging_toolkit\afni_fsl' || true
wslpath -w /mnt/d/imaging_toolkit || true

echo "=== bids ==="
if [ -d /mnt/z/MRIdata/cm_monkey_qst_bids ]; then
  echo OK_bids
else
  echo MISSING_bids
fi

echo "=== tools ==="
for c in 3dvolreg 3dAllineate 3drefit antsRegistrationSyNQuick.sh flirt python3 git gh; do
  if command -v "$c" >/dev/null 2>&1; then
    echo "OK $c=$(command -v "$c")"
  else
    echo "MISSING $c"
  fi
done

echo "=== python pkgs ==="
python3 -c 'import nibabel,numpy,pandas,scipy,matplotlib; print("python OK")' 2>&1 || echo python_pkgs_missing

echo "=== pipeline paths ==="
for p in /home/bb/abin /home/bb/ants-2.5.1/bin; do
  [ -d "$p" ] && echo "OK $p" || echo "MISSING $p"
done
