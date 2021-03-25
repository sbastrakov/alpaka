/* Copyright 2020-2021 Sergei Bastrakov
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <cstdint>


namespace alpaka
{
    namespace omp
    {
        //#############################################################################
        //! Representation of OpenMP schedule information: kind and chunk size. This class can be used regardless of
        //! whether OpenMP is enabled.
        struct Schedule
        {
            enum Kind{
                //! Default corresponding to no schedule() clause
                Default,
                Static,
                Dynamic,
                Guided,
                Auto,
                Runtime
            };

            //! Schedule kind.
            Kind kind;

            //! Chunk size. Same as in OpenMP, value 0 corresponds to default chunk size. Using int and not a
            //! fixed-width type to match OpenMP API.
            int chunkSize;

            //! The provided value myKind has to be supported by the underlying OpenMP implementation.
            //! It does not have to be one of the constants defined above.
            //! A default-constructed schedule does not affect internal control variables of OpenMP.
            //! The constructor is constexpr to simplify creation of static constexpr ompSchedule in user code.
            ALPAKA_FN_HOST constexpr Schedule(Kind myKind = Default, int myChunkSize = 0)
                : kind(myKind)
                , chunkSize(myChunkSize)
            {
            }
        };

    } // namespace omp
} // namespace alpaka
