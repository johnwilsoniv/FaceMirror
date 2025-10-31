#!/usr/bin/env python3
"""
Add iostream include to Face_utils.cpp for debug output
"""

cpp_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/src/Face_utils.cpp"

# Read the file
with open(cpp_file, 'r', encoding='latin-1') as f:
    content = f.read()

# Add iostream include after other includes
if '#include <iostream>' not in content:
    content = content.replace(
        '#include <RotationHelpers.h>',
        '#include <RotationHelpers.h>\n#include <iostream>'
    )

    # Write back
    with open(cpp_file, 'w', encoding='latin-1') as f:
        f.write(content)
    print("✓ Added #include <iostream> to Face_utils.cpp")
else:
    print("iostream already included")
