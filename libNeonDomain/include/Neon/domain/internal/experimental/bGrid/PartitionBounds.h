#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/internal/experimental/bGrid/SpanClassifier.h"

namespace Neon::domain::internal::experimental::bGrid::details {


class PartitionBounds
{
    struct Bounds
    {
        int first;
        int count;
    };
    struct GhostTarget
    {
        Neon::SetIdx setIdx;
        ByDirection  byDirection;
    };
    // -------------------------------------
    // | Internal  | BoundaryUP| BoundaryDW | Ghost UP     |  Ghost Dw    |
    // | Bulk | Bc | Bulk | Bc | Bulk | Bc  | Bulk | Bc    | Bulk | Bc    |
    // |           |           |            | Ghost setIdx | Ghost setIdx |
    // -------------------------------------
    PartitionBounds() = default;

    PartitionBounds(Neon::Backend const&  backend,
                    SpanClassifier const& span);

    auto getBoundsInternal(SetIdx) -> Bounds;
    auto getBoundsInternal(SetIdx, ByDomain) -> Bounds;

    auto getBoundsBoundary(SetIdx, ByDirection) -> Bounds;
    auto getBoundsBoundary(SetIdx, ByDirection, ByDomain) -> Bounds;

    auto getGhostBoundary(SetIdx, ByDirection) -> Bounds;
    auto getGhostBoundary(SetIdx, ByDirection, ByDomain) -> Bounds;
    auto getGhostTarget(SetIdx, ByDirection) -> GhostTarget;

   private:
    struct InfoByPartition
    {
        struct Info
        {


           private:
            Bounds      mByDomain[2];
            GhostTarget ghost;

           public:
            auto operator()(ByDomain byDomain)
                -> Bounds&
            {
                return mByDomain[static_cast<int>(byDomain)];
            }
            auto operator()(ByDomain byDomain) const
                -> Bounds const&
            {
                return mByDomain[static_cast<int>(byDomain)];
            }
            auto getGhost() -> GhostTarget&
            {
                return ghost;
            }
        };

       public:
        auto getInternal() -> Info&
        {
            return internal;
        }
        auto getBoundary(ByDirection byDirection) -> Info&
        {
            return boundary[static_cast<int>(byDirection)];
        }
        auto getGhost(ByDirection byDirection) -> Info&
        {
            return ghost[static_cast<int>(byDirection)];
        }

       private:
        Info internal;
        Info boundary[2];
        Info ghost[2];
    };

   private:
    auto getTargetGhost(Neon::SetIdx setIdx,
                        ByDirection  direction) -> GhostTarget
    {
        int         offset = direction == ByDirection::up ? 1 : -1;
        int         ngh = (setIdx.idx() + countXpu + offset) % countXpu;
        GhostTarget result;
        result.setIdx = ngh;
        result.byDirection = direction == ByDirection::up ? ByDirection::down : ByDirection::up;
        return result;
    }

    Neon::set::DataSet<InfoByPartition> mDataByPartition;
    int                                 countXpu;
};

}  // namespace Neon::domain::internal::experimental::bGrid::details
