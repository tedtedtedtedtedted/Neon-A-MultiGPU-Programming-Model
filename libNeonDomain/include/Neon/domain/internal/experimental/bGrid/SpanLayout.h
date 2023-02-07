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

    SpanLayout(
        Neon::Backend const&     backend,
        SpanDecomposition const& spanPartitioner,
        SpanClassifier const&    spanClassifier);

    auto getBoundsInternal(SetIdx)
        const -> Bounds;

    auto getBoundsInternal(SetIdx, ByDomain)
        const -> Bounds;

    auto getBoundsBoundary(SetIdx, ByDirection)
        const -> Bounds;

    auto getBoundsBoundary(
        SetIdx,
        ByDirection,
        ByDomain)
        const -> Bounds;

    auto getGhostBoundary(
        SetIdx,
        ByDirection) const -> Bounds;

    auto getGhostBoundary(
        SetIdx,
        ByDirection,
        ByDomain) const -> Bounds;

    auto getGhostTarget(
        SetIdx,
        ByDirection) const -> GhostTarget;

    auto getLocalPointOffset(
        SetIdx                setIdx,
        Neon::int32_3d const& point)
        const -> std::pair<bool, int32_t>;

    auto findPossiblyLocalPointOffset(
        SetIdx          setIdx,
        const int32_3d& point)
        const -> std::tuple<bool, int32_t, ByPartition, ByDirection, ByDomain>;

    auto findNeighbourOfInternalPoint(
        SetIdx                setIdx,
        const Neon::int32_3d& point,
        const Neon::int32_3d& offset)
        const -> std::pair<bool, int32_t>;


    auto findNeighbourOfBoundaryPoint(
        SetIdx                setIdx,
        const Neon::int32_3d& point,
        const Neon::int32_3d& nghOffset)
        const -> std::pair<bool, int32_t>;

    auto allocateBlockOriginMemSet(
        Neon::Backend const& backend,
        int                  stream)
        const -> Neon::set::MemSet_t<Neon::int32_3d>;

    auto allocateStencilRelativeIndexMap(
        const Backend&               backend,
        int                          stream,
        const Neon::domain::Stencil& stencil)
        const -> Neon::set::MemSet_t<int8_3d>;

    auto allocateBlockConnectivityMemSet(
        Neon::Backend const& backend,
        int                  stream)
        const -> Neon::set::MemSet_t<uint32_t>;

    template <typename ActiveCellLambda>
    auto allocateActiveMaskMemSet(
        const Backend&          backend,
        int                     stream,
        const ActiveCellLambda& activeCellLambda,
        const Neon::int32_3d&   domainSize,
        int                     blockSize,
        const int               discreteVoxelSpacing)
        const -> Neon::set::MemSet_t<uint32_t>;

   private:
    /**
     * Returns the firs index of the selected partition of the partition logical span
     */
    auto getClassificationOffset(
        Neon::SetIdx,
        ByPartition,
        ByDirection,
        ByDomain)
        const -> int32_t;

    auto getTargetGhost(
        Neon::SetIdx setIdx,
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
    SpanDecomposition const*            mSpanPartitioner;
    Neon::MemoryOptions                 mMemOptionsAoS;
};

template <typename ActiveCellLambda>
auto SpanLayout::allocateActiveMaskMemSet(
    const Backend&          backend,
    int                     stream,
    const ActiveCellLambda& activeCellLambda,
    const Neon::int32_3d&   domainSize,
    int                     blockSize,
    const int               discreteVoxelSpacing)
    const -> Neon::set::MemSet_t<uint32_t>
{

    int const countVoxelPerBlock = blockSize * blockSize * blockSize;
    int const count32bitWordPerBlock = (countVoxelPerBlock + 31) / 32;

    auto maskSize = backend.devSet().newDataSet<uint64_t>(
        [&](Neon::SetIdx setIdx,
            uint64_t&    val) {
            auto const& bounds = getGhostBoundary(setIdx, ByDirection::down);
            val =
                (bounds.first +
                 bounds.count) *
                NEON_DIVIDE_UP(blockSize * blockSize * blockSize,
                               bCell::sMaskSize);
        });

    Neon::set::MemSet_t<uint32_t> bitMask =
        backend.devSet().template newMemSet<uint32_t>(
            Neon::DataUse::IO_COMPUTE,
            1,
            mMemOptionsAoS,
            maskSize);

    maskSize.forEachSetIdx(
        [&](const Neon::SetIdx& setIdx,
            uint64_t&           size) {
            for (size_t i = 0; i < size; ++i) {
                bitMask.eRef(setIdx, int64_t(i)) = 0;
            }
        });

    // First pass to compute bit mask for partition {internal and boundary}
    backend.devSet().forEachSetIdxSeq(
        [&](Neon::SetIdx const& setIdx) {
            auto manageBlock =
                [&bitMask,
                 discreteVoxelSpacing,
                 domainSize,
                 activeCellLambda](Neon::SetIdx          setIdx,
                                   uint64_t              blockIdx,
                                   int                   blockSize,
                                   Neon::int32_3d const& blockOrigin) {
                    auto setCellActiveMask =
                        [&](bCell::Location::Integer x,
                            bCell::Location::Integer y,
                            bCell::Location::Integer z) {
                            bCell cell(x, y, z);
                            cell.mBlockID = blockIdx;
                            cell.mBlockSize = blockSize;
                            bitMask.eRef(setIdx,
                                         cell.getBlockMaskStride() + cell.getMaskLocalID(),
                                         0) |= 1 << cell.getMaskBitPosition();
                        };

                    // set active mask and child ID
                    for (bCell::Location::Integer z = 0; z < blockSize; z++) {
                        for (bCell::Location::Integer y = 0; y < blockSize; y++) {
                            for (bCell::Location::Integer x = 0; x < blockSize; x++) {

                                const Neon::int32_3d id(blockOrigin.x + x * discreteVoxelSpacing,
                                                        blockOrigin.y + y * discreteVoxelSpacing,
                                                        blockOrigin.z + z * discreteVoxelSpacing);

                                if (id < domainSize * discreteVoxelSpacing && activeCellLambda(id)) {
                                    bCell cell(x, y, z);
                                    cell.mBlockID = blockIdx;
                                    cell.mBlockSize = blockSize;

                                    if (id < domainSize * discreteVoxelSpacing && activeCellLambda(id)) {
                                        setCellActiveMask(x, y, z);
                                    }
                                }
                            }
                        }
                    }
                };

            for (auto byPartition : {ByPartition::internal}) {
                const auto byDirection = ByDirection::up;
                for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                    auto const& mapperVec = mSpanClassifierPtr->getMapper1Dto3D(
                        setIdx,
                        byPartition,
                        byDirection,
                        byDomain);

                    auto const start = this->getBoundsInternal(setIdx, byDomain).first;
                    for (uint64_t blockIdx = 0; blockIdx < mapperVec.size(); blockIdx++) {
                        auto const& blockOrigin = mapperVec[blockIdx];
                        manageBlock(setIdx,
                                    blockIdx + start,
                                    blockSize,
                                    blockOrigin);
                    }
                }
            }
            for (auto byPartition : {ByPartition::boundary}) {
                for (auto byDirection : {ByDirection::up, ByDirection::down}) {
                    for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                        auto const& mapperVec = mSpanClassifierPtr->getMapper1Dto3D(
                            setIdx,
                            byPartition,
                            byDirection,
                            byDomain);

                        auto const start = this->getBoundsBoundary(setIdx, byDirection, byDomain).first;
                        for (uint64_t blockIdx = 0; blockIdx < mapperVec.size(); blockIdx++) {
                            auto const& blockOrigin = mapperVec[blockIdx];
                            manageBlock(setIdx,
                                        blockIdx + start,
                                        blockSize,
                                        blockOrigin);
                        }
                    }
                }
            }
        });

    // Second pass to compute bit mask for partition {ghost}
    backend.devSet().forEachSetIdxSeq(
        [&](Neon::SetIdx const& centerSetIdx) {
            for (int setIdxOffset : {-1, 1}) {
                Neon::SetIdx nghSetIdx = (centerSetIdx + setIdxOffset + mCountXpu) % mCountXpu;
                for (auto byPartition : {ByPartition::boundary}) {
                    // Direction from the neighbour to the target
                    for (auto byDirection : {ByDirection::up, ByDirection::down}) {
                        if (setIdxOffset == +1 && byDirection == ByDirection::up)
                            continue;
                        if (setIdxOffset == -11 && byDirection == ByDirection::down)
                            continue;

                        for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                            auto const& nghMapperVec = mSpanClassifierPtr->getMapper1Dto3D(
                                nghSetIdx,
                                byPartition,
                                byDirection,
                                byDomain);

                            auto const nghStart = this->getBoundsBoundary(nghSetIdx, byDirection, byDomain).first;
                            auto const centerStart = this->getGhostBoundary(centerSetIdx,
                                                                            byDirection == ByDirection::up ? ByDirection::down : ByDirection::up,
                                                                            byDomain)
                                                         .first;

                            for (uint64_t nghblockIdxLocal = 0; nghblockIdxLocal < nghMapperVec.size(); nghblockIdxLocal++) {
                                uint64_t const nghblockIdx = nghStart + nghblockIdxLocal;
                                uint64_t const centerblockIdx = centerStart + nghblockIdxLocal;

                                // set active mask and child ID
                                for (bCell::Location::Integer z = 0; z < blockSize; z++) {
                                    for (bCell::Location::Integer y = 0; y < blockSize; y++) {
                                        for (bCell::Location::Integer x = 0; x < blockSize; x++) {

                                            bCell nghCell(x, y, z);
                                            nghCell.mBlockID = nghblockIdx;
                                            nghCell.mBlockSize = blockSize;

                                            bCell centerCell(x, y, z);
                                            centerCell.mBlockID = centerblockIdx;
                                            centerCell.mBlockSize = blockSize;

                                            bool isActive = bitMask.eVal(nghSetIdx, nghCell.getBlockMaskStride() + nghCell.getMaskLocalID(), 0) & 1 << nghCell.getMaskBitPosition();
                                            if (isActive) {
                                                bitMask.eRef(centerSetIdx, centerCell.getBlockMaskStride() + centerCell.getMaskLocalID(), 0) |= 1 << centerCell.getMaskBitPosition();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

    bitMask.updateCompute(backend, stream);
    return bitMask;
}


}  // namespace Neon::domain::internal::experimental::bGrid::details
