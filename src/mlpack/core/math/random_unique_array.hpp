/**
 * @file random_unique_array.hpp
 * @author Ryan Curtin
 *
 * This implements a utility function to fill an array with random elements
 * without replacement.
 */
#ifndef __MLPACK_CORE_MATH_RANDOM_UNIQUE_ARRAY_HPP
#define __MLPACK_CORE_MATH_RANDOM_UNIQUE_ARRAY_HPP

#include "random.hpp"

namespace mlpack {
namespace math {

/**
 * Fill the given Col<size_t> with random unique indices in the range from min
 * to max (excluding max).  The distribution used here is uniform.
 *
 * This uses an algorithm for sampling which is attributed to Robert Floyd:
 *
 * @code
 * @article{bentley1987programming,
 *   title={Programming pearls: a sample of brilliance},
 *   author={Bentley, Jon and Floyd, Bob},
 *   journal={Communications of the ACM},
 *   volume={30},
 *   number={9},
 *   pages={754--757},
 *   year={1987},
 *   publisher={ACM}
 * }
 * @endcode
 *
 * @param min Minimum element to return.
 * @param max Maximum element to return.
 * @param numElem Number of elements to return.
 * @param output Array to store unique output in.
 */
inline void RandomUniqueArray(const size_t min,
                              const size_t max,
                              const size_t numElem,
                              arma::Col<size_t>& output)
{
  output.set_size(numElem);
  output[0] = math::RandInt(min, max - numElem + 1);
  for (size_t i = 0; i < numElem; ++i)
  {
    output[i] = math::RandInt(min, max - numElem + i + 1);
    // Check previous values; switch if necessary.
    for (size_t j = 0; j < i; ++j)
      if (output[i] == output[j])
        output[i] = max - numElem + i;
  }
}

} // namespace math
} // namespace mlpack

#endif
