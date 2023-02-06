#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/internal/experimental/bGrid/SpanClassifier.h"

namespace Neon::domain::internal::experimental::bGrid::details {


class SpanLayout
{
   public:
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
    SpanLayout() = default;

    SpanLayout(Neon::Backend const&   backend,
               SpanDecomposition const& spanPartitioner,
                  SpanClassifier const&  spanClassifier);

    auto getBoundsInternal(SetIdx) const -> Bounds;
    auto getBoundsInternal(SetIdx, ByDomain) const -> Bounds;

    auto getBoundsBoundary(SetIdx, ByDirection) const -> Bounds;
    auto getBoundsBoundary(SetIdx, ByDirection, ByDomain) const -> Bounds;

    auto getGhostBoundary(SetIdx, ByDirection) const -> Bounds;
    auto getGhostBoundary(SetIdx, ByDirection, ByDomain) const -> Bounds;
    auto getGhostTarget(SetIdx, ByDirection) const -> GhostTarget;

    auto getLocalPointOffset(SetIdx setIdx, Neon::int32_3d const& point)
        const -> int32_t;

    auto getPossiblyLocalPointOffset(SetIdx setIdx, const int32_3d& point)
        const -> std::tuple<bool, int32_t, ByPartition, ByDirection, ByDomain>;


    auto getNeighbourOfInternalPoint(SetIdx                setIdx,
                                     Neon::int32_3d const& point,
                                     Neon::int32_3d const& offset)
        const -> int32_t;


    auto getNeighbourOfBoundaryPoint(SetIdx                setIdx,
                                     Neon::int32_3d const& point,
                                     Neon::int32_3d const& nghOffset)
        const -> int32_t;

    auto allocateBlockOriginMemSet(Neon::Backend const& backend, int stream)
        const ->  Neon::set::MemSet_t<Neon::int32_3d>;

   private:
    /**
     * Returns the firs index of the selected partition of the partition logical span
     */
    auto getClassificationOffset(Neon::SetIdx, ByPartition, ByDirection, ByDomain)
        const -> int32_t;

    auto getTargetGhost(Neon::SetIdx setIdx,
                        ByDirection  direction) -> GhostTarget
    {
        int         offset = direction == ByDirection::up ? 1 : -1;
        int         ngh = (setIdx.idx() + mCountXpu + offset) % mCountXpu;
        GhostTarget result;
        result.setIdx = ngh;
        result.byDirection = direction == ByDirection::up ? ByDirection::down : ByDirection::up;
        return result;
    }

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

            auto getGhost() const -> GhostTarget const&
            {
                return ghost;
            }

            auto getGhost() -> GhostTarget&
            {
                return ghost;
            }
        };

       public:
        auto getInternal()
            const -> Info const&
        {
            return internal;
        }

        auto getBoundary(ByDirection byDirection) const
            -> Info const&
        {
            return boundary[static_cast<int>(byDirection)];
        }

        auto getGhost(ByDirection byDirection)
            const -> Info const&
        {
            return ghost[static_cast<int>(byDirection)];
        }

        auto getInternal()
            -> Info&
        {
            return internal;
        }

        auto getBoundary(ByDirection byDirection)
            -> Info&
        {
            return boundary[static_cast<int>(byDirection)];
        }

        auto getGhost(ByDirection byDirection)
            -> Info&
        {
            return ghost[static_cast<int>(byDirection)];
        }

       private:
        Info internal;
        Info boundary[2];
        Info ghost[2];
    };

    Neon::set::DataSet<InfoByPartition> mDataByPartition;
    int                                 mCountXpu;
    SpanClassifier const*               mSpanClassifierPtr;
    SpanDecomposition const*              mSpanPartitioner;
};

}  // namespace Neon::domain::internal::experimental::bGrid::details
