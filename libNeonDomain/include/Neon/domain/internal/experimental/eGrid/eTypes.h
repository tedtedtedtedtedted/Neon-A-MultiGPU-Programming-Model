#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::experimental::eGrid::details {
using Span3DIndex = Neon::uint32_3d;
using MemIdx = uint32_t;
using Ngh3DOffset = Neon::int8_3d;
using LogicalIdx= uint32_t /** type associated with a 1D local index */;
}  // namespace Neon::domain::internal::experimental::eGrid::details

#include "Neon/domain/internal/bGrid/bCell_imp.h"