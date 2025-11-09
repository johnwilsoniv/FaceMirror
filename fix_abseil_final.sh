#!/bin/bash
cd /opt/homebrew/opt/abseil/lib
rm -f libabsl_*.2407.0.0.dylib
for lib in libabsl_*.2508.0.0.dylib; do
    target="${lib/2508.0.0/2407.0.0}"
    ln -s "$lib" "$target"
done
echo "Created $(ls -1 libabsl_*.2407.0.0.dylib 2>/dev/null | wc -l) symlinks"
