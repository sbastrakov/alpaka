/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

//#############################################################################
// Base kernel, not to be used directly in unit tests
struct KernelWithOmpScheduleBase
{
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        alpaka::ignore_unused(acc);
        // Currently only test that everything compiles.
        // TODO: if we are on the Omp2Blocks accelerator and have OpenMP >= 3.0,
        // check that the current schedule is set to ompSchedule
        ALPAKA_CHECK(*success, true);
    }
};

// Schedule to be used by all kernels in this file
static constexpr auto expectedSchedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Dynamic, 10};

// Kernel that sets the schedule via constexpr ompSchedule.
// Checks that this variable is only declared and not defined, It also tests that
// alpaka never odr-uses it.
struct KernelWithConstexprMemberOmpSchedule : KernelWithOmpScheduleBase
{
    static constexpr auto ompSchedule = expectedSchedule;
};

// Kernel that sets the schedule via non-constexpr ompSchedule.
struct KernelWithMemberOmpSchedule : KernelWithOmpScheduleBase
{
    static const alpaka::omp::Schedule ompSchedule;
};
// In this case, the member has to be defined externally
const alpaka::omp::Schedule KernelWithMemberOmpSchedule::ompSchedule = expectedSchedule;

// Kernel that sets the schedule via partial specialization of a trait
struct KernelWithTraitOmpSchedule : KernelWithOmpScheduleBase
{
};

namespace alpaka
{
    namespace traits
    {
        template<typename TAcc>
        struct OmpSchedule<KernelWithTraitOmpSchedule, TAcc>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDim, typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto getOmpSchedule(
                KernelWithTraitOmpSchedule const& kernelFnObj,
                Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
                Vec<TDim, Idx<TAcc>> const& threadElemExtent,
                TArgs const&... args) -> alpaka::omp::Schedule
            {
                alpaka::ignore_unused(kernelFnObj);
                alpaka::ignore_unused(blockThreadExtent);
                alpaka::ignore_unused(threadElemExtent);
                alpaka::ignore_unused(args...);

                return expectedSchedule;
            }
        };
    } // namespace traits
} // namespace alpaka

// Generic testing routine for the given kernel type
template<typename TAcc, typename TKernel>
void test()
{
    using Acc = TAcc;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    TKernel kernel;

    REQUIRE(fixture(kernel));
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("kernelWithConstexprMemberOmpSchedule", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithConstexprMemberOmpSchedule>();
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("kernelWithMemberOmpSchedule", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithMemberOmpSchedule>();
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("kernelWithTraitOmpSchedule", "[kernel]", alpaka::test::TestAccs)
{
    test<TestType, KernelWithTraitOmpSchedule>();
}
