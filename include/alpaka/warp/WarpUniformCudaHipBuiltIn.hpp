/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/core/Unused.hpp>
#include <alpaka/warp/Traits.hpp>

#include <cstdint>

namespace alpaka
{
    namespace warp
    {
        //#############################################################################
        //! The GPU CUDA/HIP warp.
        class WarpUniformCudaHipBuiltIn : public concepts::Implements<ConceptWarp, WarpUniformCudaHipBuiltIn>
        {
        public:
            //-----------------------------------------------------------------------------
            WarpUniformCudaHipBuiltIn() = default;
            //-----------------------------------------------------------------------------
            __device__ WarpUniformCudaHipBuiltIn(WarpUniformCudaHipBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ WarpUniformCudaHipBuiltIn(WarpUniformCudaHipBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(WarpUniformCudaHipBuiltIn const &) -> WarpUniformCudaHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(WarpUniformCudaHipBuiltIn &&) -> WarpUniformCudaHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            ~WarpUniformCudaHipBuiltIn() = default;
        };

        namespace traits
        {
            //#############################################################################
            template<>
            struct GetSize<
                WarpUniformCudaHipBuiltIn>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto getSize(
                    warp::WarpUniformCudaHipBuiltIn const & /*warp*/)
                -> std::int32_t
                {
                    return warpSize;
                }
            };

            //#############################################################################
            template<>
            struct All<
                WarpUniformCudaHipBuiltIn>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto all(
                    warp::WarpUniformCudaHipBuiltIn const & warp,
                    std::int32_t predicate)
                -> std::int32_t
                {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    auto const mask = static_cast<std::uint32_t>(activemask(warp));
                    return __all_sync(
                        mask,
                        predicate);
#else
                    ignore_unused(warp);
                    return __all(predicate);
#endif
                }
            };

            //#############################################################################
            template<>
            struct Any<
                WarpUniformCudaHipBuiltIn>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto any(
                    warp::WarpUniformCudaHipBuiltIn const & warp,
                    std::int32_t predicate)
                -> std::int32_t
                {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    auto const mask = static_cast<std::uint32_t>(activemask(warp));
                    return __any_sync(
                        mask,
                        predicate);
#else
                    ignore_unused(warp);
                    return __any(predicate);
#endif
                }
            };

            //#############################################################################
            template<>
            struct Ballot<
                WarpUniformCudaHipBuiltIn>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto ballot(
                    warp::WarpUniformCudaHipBuiltIn const & warp,
                    std::int32_t predicate)
                -> std::uint64_t
                {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    auto const mask = static_cast<std::uint32_t>(activemask(warp));
                    return __ballot_sync(
                        mask,
                        predicate);
#else
                    ignore_unused(warp);
                    return __ballot(predicate);
#endif
                }
            };

            //#############################################################################
            template<>
            struct Activemask<
                WarpUniformCudaHipBuiltIn>
            {
                //-----------------------------------------------------------------------------
                __device__ static auto activemask(
                    warp::WarpUniformCudaHipBuiltIn const & /*warp*/)
                -> std::uint64_t
                {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    // Workaround for clang + CUDA 9.2 which uses the wrong PTX ISA,
                    // discussion in https://github.com/alpaka-group/alpaka/pull/1003
                    // Can't use __activemask(), so emulate with __ballot_sync()
    #if BOOST_COMP_CLANG_CUDA && BOOST_LANG_CUDA == BOOST_VERSION_NUMBER(9, 2, 0)
                    return static_cast<std::uint64_t>(__ballot_sync(
                        0xffffffff,
                        1));
    #else
                    return static_cast<std::uint64_t>(__activemask());
    #endif
#else
                    // No HIP intrinsic for it, emulate via ballot
                    return __ballot(1);
#endif
                }
            };
        }
    }
}

#endif
