#!/bin/bash
# Build the GitHub-fork flash-train (feat/kda working tree) inside tanbo_mega_bw52 -> /public/home/tanbo/xplat/ft_site_gh
set -u
export DTKROOT=/opt/dtk
SRC=/public/home/tanbo/ft_kda_gh; W=/public/home/tanbo/xplat/ftgh_build; NEW=${1:-/public/home/tanbo/xplat/ft_site_gh}
mkdir -p $W && rsync -a --delete --exclude .git --exclude build-wheel $SRC/ $W/
cd $W && rm -rf /tmp/dist_gh
if ! python3 -m pip wheel . --no-deps -w /tmp/dist_gh > /tmp/build_gh.log 2>&1; then echo BUILD_FAILED; grep -m20 -E "error" /tmp/build_gh.log; tail -5 /tmp/build_gh.log; exit 1; fi
rm -rf $NEW && python3 -m pip install --no-deps --target $NEW /tmp/dist_gh/flash_train-*.whl >> /tmp/build_gh.log 2>&1 || { echo INSTALL_FAILED; exit 1; }
cp /tmp/dist_gh/flash_train-*.whl /public/home/tanbo/xplat/ 
echo "BUILD_OK $(md5sum $NEW/flash_train/libflash_train.so 2>/dev/null | cut -c1-12) $(ls $NEW/flash_train/torch)"
