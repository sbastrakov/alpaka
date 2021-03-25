/* Copyright 2020-2021 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/core/Unused.hpp>

#include <catch2/catch.hpp>

//-----------------------------------------------------------------------------
TEST_CASE("ompScheduleDefaultConstructor", "[core]")
{
    auto const schedule = alpaka::omp::Schedule{};
    alpaka::ignore_unused(schedule);
}

//-----------------------------------------------------------------------------
TEST_CASE("ompScheduleConstructor", "[core]")
{
    auto const staticSchedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Static, 5};
    alpaka::ignore_unused(staticSchedule);

    auto const guidedSchedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Guided};
    alpaka::ignore_unused(guidedSchedule);
}

//-----------------------------------------------------------------------------
TEST_CASE("ompScheduleConstexprConstructor", "[core]")
{
    constexpr auto schedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Dynamic};
    alpaka::ignore_unused(schedule);
}
