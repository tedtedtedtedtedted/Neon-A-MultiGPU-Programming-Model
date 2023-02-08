#include "Neon/domain/internal/experimental/eGrid/eCell.h"

namespace Neon::domain::internal::experimental::eGrid::details {

NEON_CUDA_HOST_DEVICE inline eCell::eCell(
    LogicalIdx const& location)
{
    mLogicalIdx = location;
}

NEON_CUDA_HOST_DEVICE inline auto eCell::set() -> LogicalIdx&
{
    return mLogicalIdx;
}
NEON_CUDA_HOST_DEVICE inline auto eCell::get()
    const -> const LogicalIdx&
{
    return mLogicalIdx;
}

}  // namespace Neon::domain::internal::experimental::eGrid::details