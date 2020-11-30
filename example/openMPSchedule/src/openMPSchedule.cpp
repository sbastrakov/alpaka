/* Copyright 2019-2020 Benjamin Worpitz, Erik Zenker, Sergei Bastrakov
 *
 * This file exemplifies usage of alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

// This example only makes sense with alpaka AccCpuOmp2Blocks backend enabled
// and OpenMP runtime supporting at least 3.0. Disable it for other cases.
#if defined _OPENMP && _OPENMP >= 200805 && ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    include <alpaka/alpaka.hpp>
#    include <alpaka/example/ExampleDefaultAcc.hpp>

#    include <cstdint>
#    include <iostream>

//#############################################################################
//! OpenMP schedule demonstration kernel
//!
//! Demonstrates how to set OpenMP schedule and prints distribution of alpaka
//! thread indices between OpenMP threads
struct OpenMPScheduleKernel
{
    // A simple way to set a schedule for all invocations on this kernel
    // using a constexpr member. Note that constexpr is not required,
    // however otherwise there has to be an external definition.
    // Another way to set the schedule is via trait specialization, as
    // demonstrated after the kernel body. Trait specialization is checked first,
    // so the member only has effect when the trait is not specialized for this type.
    static constexpr auto ompSchedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Static, 1};

    //-----------------------------------------------------------------------------
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        // For simplicity assume 1d index space
        using Idx = alpaka::Idx<TAcc>;
        Idx const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Check that the current schedule used by OpenMP matches ompSchedule
        if(globalThreadIdx == 0)
        {
            omp_sched_t kind;
            int actualChunkSize = 0;
            omp_get_schedule(&kind, &actualChunkSize);
            auto const actualKind = static_cast<uint32_t>(kind);
            if(ompSchedule.kind != actualKind)
                printf("ERROR: expected OpenMP schedule kind %u, actual %u\n", ompSchedule.kind, actualKind);
            if(ompSchedule.chunkSize != actualChunkSize)
                printf(
                    "ERROR: expected OpenMP schedule chunk size %d, actual %d\n",
                    ompSchedule.chunkSize,
                    actualChunkSize);
        }

        // Print work distribution between threads for illustration
        printf(
            "alpaka global thread index %u is processed by OpenMP thread %d\n",
            static_cast<uint32_t>(globalThreadIdx),
            omp_get_thread_num());
    }
};


namespace alpaka
{
    namespace traits
    {
        //! Schedule trait specialization for our kernel.
        //! When uncommented, that would have priority over the ompSchedule
        //! member. This is the most general way to define a schedule, while
        //! that member is a shortcut for the common case.
        /*
        template<typename TAcc>
        struct OmpSchedule<OpenMPScheduleKernel, TAcc>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TDim, typename... TArgs>
            ALPAKA_FN_HOST_ACC static auto getOmpSchedule(
                OpenMPScheduleKernel const& kernelFnObj,
                Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
                Vec<TDim, Idx<TAcc>> const& threadElemExtent,
                TArgs const&... args) -> alpaka::omp::Schedule
            {
                // Determine schedule at runtime for the given kernel and
                // run parameters. For this particular example kernel, TArgs is
                // an empty pack and can be removed.
                alpaka::ignore_unused(kernelFnObj);
                alpaka::ignore_unused(blockThreadExtent);
                alpaka::ignore_unused(threadElemExtent);
                alpaka::ignore_unused(args...);

                return alpaka::omp::Schedule{ alpaka::omp::Schedule::Static, 1 };
            }
        };
        */
    } // namespace traits
} // namespace alpaka

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#    if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#    else
    using Idx = std::size_t;

    // OpenMP schedule illustrated by this example only has effect with
    // with the AccCpuOmp2Blocks accelerator.
    // This example also assumes 1d for simplicity.
    using Acc = alpaka::AccCpuOmp2Blocks<alpaka::DimInt<1>, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    using QueueProperty = alpaka::Blocking;
    using Queue = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    Idx const threadsPerGrid = 30u;
    Idx const elementsPerThread = 1u;
    auto const workDiv = alpaka::getValidWorkDiv<Acc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    // Instantiate the kernel function object
    OpenMPScheduleKernel kernel;

    // Run the kernel
    alpaka::exec<Acc>(queue, workDiv, kernel);
    alpaka::wait(queue);

    return EXIT_SUCCESS;
#    endif
}

#endif
