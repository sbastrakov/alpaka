/* Copyright 2020 Sergei Bastrakov
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
        //! Representation of OpenMP schedule information: kind and chunk size.
        //! This class can be used regardless of whether OpenMP is enabled.
        struct Schedule
        {
            //#############################################################################
            //! Integers corresponding to the mandatory OpenMP omp_sched_t enum
            //! values as of version 5.0.
            constexpr static uint32_t Static = 1;
            constexpr static uint32_t Dynamic = 2;
            constexpr static uint32_t Guided = 3;
            constexpr static uint32_t Auto = 4;
            // Each schedule value can be combined with monotonic using + or |
            constexpr static uint32_t Monotonic = 0x80000000u;

            //! Schedule kind.
            //!
            //! We cannot simply use type omp_sched_t since this struct is
            //! agnostic of OpenMP. We also cannot create an own mirror enum,
            //! since OpenMP implementations are allowed to extend the range of
            //! values beyond the standard ones defined above. So we have to
            //! accept and store any uint32_t value, and for non-standard values
            //! a user has to ensure the underlying implementation supports it.
            uint32_t kind;

            //! Chunk size.
            //! Same as in OpenMP, value 0 corresponds to default chunk size.
            //! Using int and not a fixed-width type to match OpenMP API.
            int chunkSize;

            //! The provided value myKind has to be supported by the underlying
            //! OpenMP implementation. It does not have to be one of the
            //! constants defined above.
            //! The constructor is constexpr to simplify creation of static
            //! constexpr ompSchedule in user code.
            constexpr Schedule(uint32_t myKind = Guided, int myChunkSize = 0) : kind(myKind), chunkSize(myChunkSize)
            {
            }
        };

        //-----------------------------------------------------------------------------
        //! Get the OpenMP schedule that is applied when the runtime schedule is used.
        //!
        //! When executed without OpenMP or with OpenMP < 3.0, returns default
        //! schedule.
        //!
        //! \return Schedule object.
        ALPAKA_FN_HOST inline auto getSchedule()
        {
            // Getting a runtime schedule requires OpenMP 3.0 or newer
#if defined _OPENMP && _OPENMP >= 200805
            omp_sched_t ompKind;
            int chunkSize = 0;
            omp_get_schedule(&ompKind, &chunkSize);
            return Schedule{static_cast<uint32_t>(ompKind), chunkSize};
#else
            return Schedule{};
#endif
        }

        //-----------------------------------------------------------------------------
        //! Set the OpenMP schedule that is applied when the runtime schedule is used.
        //!
        //! When executed without OpenMP or with OpenMP < 3.0, does nothing.
        //! For OpenMP < 3.0 the schedule is controlled by OMP_SCHEDULE
        //! environment variable.
        ALPAKA_FN_HOST inline void setSchedule(Schedule schedule)
        {
#if defined _OPENMP && _OPENMP >= 200805
            omp_set_schedule(static_cast<omp_sched_t>(schedule.kind), schedule.chunkSize);
#else
            ignore_unused(schedule);
#endif
        }

    } // namespace omp
} // namespace alpaka
