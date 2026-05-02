// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
//
// VENDORED WINDOWS PATCH (FaceMirror): replaces the version of this header
// shipped in pyfhog v0.1.4 (https://github.com/johnwilsoniv/pyfhog/blob/v0.1.4/
// src/cpp/dlib/test_for_odr_violations.h). The upstream v0.1.4 commit re-enabled
// the dlib ODR/version-mismatch sentinels that v0.1.3 had commented out;
// they fail to link on Windows MSVC because pyfhog uses dlib in header-only
// mode and never compiles dlib/all/source.cpp (where the extern symbols
// are defined). On macOS/Linux they happen not to be checked the same way.
//
// We need v0.1.4's HOG indexing fix in src/cpp/fhog_wrapper.cpp (the
// transposed [x][y] vs [y][x] bug that caused the AU correlation collapse
// on Windows), but we don't need the link-time sentinels. So we keep them
// commented out, matching the v0.1.3 state of this one header.
#ifndef DLIB_TEST_FOR_ODR_VIOLATIONS_H_
#define DLIB_TEST_FOR_ODR_VIOLATIONS_H_

#include "assert.h"
#include "config.h"

extern "C"
{
// =========================>>> WHY YOU ARE GETTING AN ERROR HERE <<<=========================
// DISABLED FOR PYFHOG-WIN: see header banner above.
// #ifdef ENABLE_ASSERTS
//     const extern int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1;
//     const int DLIB_NO_WARN_UNUSED dlib_check_assert_helper_variable = USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1;
// #else
//     const extern int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_;
//     const int DLIB_NO_WARN_UNUSED dlib_check_assert_helper_variable = USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_;
// #endif



// The point of this block of code is to cause a link time error if someone builds dlib via
// cmake as a separately installable library, and therefore generates a dlib/config.h from
// cmake, but then proceeds to use the default unconfigured dlib/config.h from version
// control.  It should be obvious why this is bad, if it isn't you need to read a book
// about C++.  Moreover, it can only happen if someone manually copies files around and
// messes things up.  If instead they run `make install` or `cmake --build .  --target
// install` things will be setup correctly, which is what they should do.  To summarize: DO
// NOT BUILD A STANDALONE DLIB AND THEN GO CHERRY PICKING FILES FROM THE BUILD FOLDER AND
// MIXING THEM WITH THE SOURCE FROM GITHUB.  USE CMAKE'S INSTALL SCRIPTS TO INSTALL DLIB.
// Or even better, don't install dlib at all and instead build your program as shown in
// examples/CMakeLists.txt
#if defined(DLIB_NOT_CONFIGURED) && !defined(DLIB__CMAKE_GENERATED_A_CONFIG_H_FILE)
    const extern int USER_ERROR__inconsistent_build_configuration__see_dlib_faq_2;
    const int DLIB_NO_WARN_UNUSED dlib_check_not_configured_helper_variable = USER_ERROR__inconsistent_build_configuration__see_dlib_faq_2;
#endif



// Cause the user to get a linker error if they try to use header files from one version of
// dlib with the compiled binary from a different version of dlib.
// DISABLED FOR PYFHOG-WIN: same Windows linker issue as the ODR sentinel above.
// #ifdef DLIB_CHECK_FOR_VERSION_MISMATCH
//     const extern int DLIB_CHECK_FOR_VERSION_MISMATCH;
//     const int DLIB_NO_WARN_UNUSED dlib_check_for_version_mismatch = DLIB_CHECK_FOR_VERSION_MISMATCH;
// #endif

}

#endif // DLIB_TEST_FOR_ODR_VIOLATIONS_H_

