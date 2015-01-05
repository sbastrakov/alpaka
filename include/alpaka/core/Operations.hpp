/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Vec.hpp>  // alpaka::vec

#include <algorithm>            // std::min, std::max

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! Defines operation functors.
    //-----------------------------------------------------------------------------
    namespace operations
    {
        //#############################################################################
        //! The addition functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Add
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref += value;
                return old;
            }
        };
        //#############################################################################
        //! The subtraction functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Sub
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref -= value;
                return old;
            }
        };
        //#############################################################################
        //! The minimum functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Min
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref = std::min(ref, value);
                return old;
            }
        };
        //#############################################################################
        //! The maximum functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Max
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref = std::max(ref, value);
                return old;
            }
        };
        //#############################################################################
        //! The exchange functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Exch
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref = value;
                return old;
            }
        };
        //#############################################################################
        //! The increment functor.
        //!
        //! Increments up to value, then reset to 0.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Inc
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref = ((old >= value) ? 0 : old+1);
                return old;
            }
        };
        //#############################################################################
        //! The decrement functor.
        //!
        //! Decrement down to 0, then reset to value.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Dec
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref = (((old == 0) || (old > value)) ? value : (old - 1));
                return old;
            }
        };
        //#############################################################################
        //! The and functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct And
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref &= value;
                return old;
            }
        };
        //#############################################################################
        //! The or functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Or
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref |= value;
                return old;
            }
        };
        //#############################################################################
        //! The exclusive or functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Xor
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * const addr, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref ^= value;
                return old;
            }
        };
        //#############################################################################
        //! The compare and swap functor.
        //!
        //! \return The old value of addr.
        //#############################################################################
        struct Cas
        {
            template<typename T>
            ALPAKA_FCT_HOST_ACC T operator()(T * addr, T const & compare, T const & value) const
            {
                auto const old(*addr);
                auto & ref(*addr);
                ref = ((old == compare) ? value : old);
                return old;
            }
        };
    }

    using namespace operations;
}