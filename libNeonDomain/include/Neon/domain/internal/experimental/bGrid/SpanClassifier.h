#pragma once
#include "Neon/core/core.h"

#include "Neon/set/Containter.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"

#include "Neon/domain/internal/experimental/bGrid/SpanPartitioner.h"

#include "Neon/domain/patterns/PatternScalar.h"

#include "Neon/domain/tools/IndexSpaceTable.h"
#include "Neon/domain/tools/PointHashTable.h"

namespace Neon::domain::internal::experimental::bGrid {

template <typename T, int C>
class bField;

class SpanClassifier
{
    SpanClassifier() = default;

    template <typename ActiveCellLambda,
              typename Block3dIdxToBlockOrigin,
              typename GetVoxelAbsolute3DIdx>
    SpanClassifier(const Neon::Backend&           backend,
                   const ActiveCellLambda&        activeCellLambda,
                   const Block3dIdxToBlockOrigin& block3dIdxToBlockOrigin,
                   const GetVoxelAbsolute3DIdx&   getVoxelAbsolute3DIdx,
                   const Neon::int32_3d&          block3DSpan,
                   const int&                     blockSize,
                   const Neon::int32_3d&          domainSize,
                   const int&                     discreteVoxelSpacing,
                   const SpanPartitioner&);

    enum struct ByPartition
    {
        internal = 0,
        boundary = 1
    };

    enum struct ByDomain
    {
        boundary = 0,
        bulk = 1
    };

    auto getCount(ByPartition, ByDomain) const
        -> int;

    auto get3dIdx(ByPartition, ByDomain) const
        -> const std::vector<Neon::index_3d>&;

   private:
    struct Info
    {
        std::vector<Neon::index_3d>                           id1dTo3d;
        Neon::domain::tool::PointHashTable<int32_t, uint32_t> id3dTo1d;
    };

    using datSetLeve0Type = Info;
    using datSetLeve1Type = std::array<datSetLeve0Type, 2>;
    using datSetLeve2Type = std::array<datSetLeve1Type, 2>;
    using Data = Neon::set::DataSet<datSetLeve2Type>;

    Data mData;
};

template <typename ActiveCellLambda,
          typename Block3dIdxToBlockOrigin,
          typename GetVoxelAbsolute3DIdx>
SpanClassifier::SpanClassifier(const Neon::Backend&           backend,
                               const ActiveCellLambda&        activeCellLambda,
                               const Block3dIdxToBlockOrigin& block3dIdxToBlockOrigin,
                               const GetVoxelAbsolute3DIdx&   getVoxelAbsolute3DIdx,
                               const Neon::int32_3d&          block3DSpan,
                               const int&                     blockSize,
                               const Neon::int32_3d&          domainSize,
                               const int&                     discreteVoxelSpacing,
                               const SpanPartitioner&         spanPartitioner)
{
    // For each Partition
    backend.devSet().forEachSetIdxSeq(
        [&](const Neon::SetIdx& setIdx) {
            int beginZ = spanPartitioner.getFirstZSliceIdx()[setIdx];
            int lastZ = spanPartitioner.getLastZSliceIdx()[setIdx];

            // We are running in the inner partition blocks
            for (int z = beginZ + 1; z < lastZ; z++) {
                for (int y = 0; y < block3DSpan.y; y++) {
                    for (int x = 0; x < block3DSpan.x; x++) {
                        Neon::int32_3d idx(x,y,z);

                    }
                }
            }
        });
}
}  // namespace Neon::domain::internal::experimental::bGrid
