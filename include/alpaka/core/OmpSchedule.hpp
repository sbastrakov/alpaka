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
        //!
        //! This class can be used regardless of whether OpenMP is enabled
        struct Schedule
        {
            //#############################################################################
            //! Mirror of the OpenMP enum omp_sched_t with the corresponding integer values
            enum Kind
            {
                Static = 1,
                Dynamic = 2,
                Guided = 3,
                Auto = 4
            };

            //! Integer representation of schedule kind.
            //!
            //! Not stored as Kind to reduce casts when converting to and from omp_sched_t
            uint32_t kind;

            //! Chink size.
            //!
            //! Same as in OpenMP, value 0 corresponds to default chunk size
            int chunkSize;

            constexpr Schedule(Kind kind = Guided, int chunkSize = 0)
                : kind(static_cast<uint32_t>(kind))
                , chunkSize(chunkSize)
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
            auto result = Schedule{};
            omp_sched_t kind;
            omp_get_schedule(&kind, &result.chunkSize);
            result.kind = static_cast<uint32_t>(kind);
            return result;
#else
            return Schedule{};
#endif
        }

        //-----------------------------------------------------------------------------
        //! Set the OpenMP schedule that is applied when the runtime schedule is used.
        //!
        //! When executed without OpenMP or with OpenMP < 3.0, does nothing.
        ALPAKA_FN_HOST inline void setSchedule(Schedule schedule)
        {
#if defined _OPENMP && _OPENMP >= 200805
            auto const kind = static_cast<omp_sched_t>(schedule.kind);
            omp_set_schedule(kind, schedule.chunkSize);
#else
            ignore_unused(schedule);
#endif
        }

    } // namespace omp
} // namespace alpaka
