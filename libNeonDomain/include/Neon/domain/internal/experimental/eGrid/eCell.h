#pragma once

#include "Neon/core/core.h"
#include "Neon/domain/internal/experimental/eGrid/eTypes.h"

namespace Neon::domain::internal::experimental::eGrid::details {
class eCell
{
   public:
    friend class bPartitionIndexSpace;

    using Span3DIndex = Neon::domain::internal::experimental::eGrid::details::Span3DIndex;
    using MemIndex = Neon::domain::internal::experimental::eGrid::details::MemIdx;
    using Ngh3DOffset = Neon::domain::internal::experimental::eGrid::details::Ngh3DOffset;
    using LogicalIdx = Neon::domain::internal::experimental::eGrid::details::LogicalIdx;

    friend class ePartitionIndexSpace;
    friend class eGrid;

    template <typename T, int C>
    friend class ePartition;

    template <typename T, int C>
    friend class eField;

    using OuterCell = eCell;


    eCell() = default;
    virtual ~eCell() = default;

    NEON_CUDA_HOST_DEVICE inline auto isActive() const -> bool;

    // the local index within the block
    LogicalIdx mLogicalIdx;

    NEON_CUDA_HOST_DEVICE inline explicit eCell(LogicalIdx const&  location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> LogicalIdx&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const LogicalIdx&;

};
}  // namespace Neon::domain::internal::experimental::eGrid::details

#include "Neon/domain/internal/bGrid/bCell_imp.h"