#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/internal/sGrid/sCell.h"

namespace Neon::domain::internal::sGrid {

NEON_CUDA_HOST_DEVICE inline sCell::sCell(Location location)
{
    mLocation = location;
}

NEON_CUDA_HOST_DEVICE inline auto sCell::get() -> Location&
{
    return mLocation;
}

NEON_CUDA_HOST_DEVICE inline auto sCell::get() const -> const Location&
{
    return mLocation;
}
}  // namespace Neon::domain::internal::sGrid