/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/intrinsic/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

//#############################################################################
class ThreadfenceTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        // For now just test the function is callable
        alpaka::intrinsic::threadfence(acc);
        ALPAKA_CHECK(*success, true);
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "threadfence", "[intrinsic]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());
    ThreadfenceTestKernel kernel;
    REQUIRE(
        fixture(
            kernel));
}
