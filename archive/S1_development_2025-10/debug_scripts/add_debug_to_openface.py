#!/usr/bin/env python3
"""
Add debug output to OpenFace C++ Face_utils.cpp to print transform matrices
"""

import re

cpp_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/src/Face_utils.cpp"

# Read the file (use latin-1 to handle special characters)
with open(cpp_file, 'r', encoding='latin-1') as f:
    content = f.read()

# Debug code to insert (using tabs for indentation)
debug_code = '''
\t\t// DEBUG: Print transform matrices for comparison with Python
\t\tstatic int frame_count = 0;
\t\tframe_count++;
\t\tif (frame_count == 1) {  // Print first frame only
\t\t\tstd::cout << "=== OpenFace Alignment Debug (Frame 1) ===" << std::endl;
\t\t\tstd::cout << "scale_rot_matrix:" << std::endl;
\t\t\tstd::cout << "  [[" << scale_rot_matrix(0,0) << ", " << scale_rot_matrix(0,1) << "]," << std::endl;
\t\t\tstd::cout << "   [" << scale_rot_matrix(1,0) << ", " << scale_rot_matrix(1,1) << "]]" << std::endl;
\t\t\tstd::cout << "params_global tx,ty: " << tx << ", " << ty << std::endl;
\t\t\tstd::cout << "T after transform: " << T(0) << ", " << T(1) << std::endl;
\t\t\tstd::cout << "warp_matrix:" << std::endl;
\t\t\tstd::cout << "  [[" << warp_matrix(0,0) << ", " << warp_matrix(0,1) << ", " << warp_matrix(0,2) << "]," << std::endl;
\t\t\tstd::cout << "   [" << warp_matrix(1,0) << ", " << warp_matrix(1,1) << ", " << warp_matrix(1,2) << "]]" << std::endl;
\t\t\tstd::cout << "=======================================" << std::endl;
\t\t}
'''

# Find the line with cv::warpAffine and insert debug code before it
pattern = r'(\t\twarp_matrix\(1,2\) = -T\(1\) \+ out_height/2;\n)'
replacement = r'\1' + debug_code + '\n'

modified_content = re.sub(pattern, replacement, content)

if content == modified_content:
    print("❌ Pattern not found! Could not add debug code.")
else:
    # Write back (use latin-1)
    with open(cpp_file, 'w', encoding='latin-1') as f:
        f.write(modified_content)
    print(f"✓ Added debug output to {cpp_file}")
    print("  Debug info will print for first frame only")
