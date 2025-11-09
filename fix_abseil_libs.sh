#!/bin/bash
cd /opt/homebrew/opt/abseil/lib
for lib in libabsl_*.2508.0.0.dylib; do
    target="${lib/2508.0.0/2407.0.0}"
    ln -sf "$lib" "$target"
    echo "Created: $target -> $lib"
done
