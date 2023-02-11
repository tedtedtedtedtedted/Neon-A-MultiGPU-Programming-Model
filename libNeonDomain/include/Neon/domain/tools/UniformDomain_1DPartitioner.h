#pragma once

#include "Neon/domain/tools/partitioning/Cassifications.h"
#include "Neon/domain/tools/partitioning/SpanClassifier.h"
#include "Neon/domain/tools/partitioning/SpanDecomposition.h"
#include "Neon/domain/tools/partitioning/SpanLayout.h"

namespace Neon::domain::tools::partitioning {

class UniformDomain_1DPartitioner
{
   public:
    UniformDomain_1DPartitioner() = default;

    template <typename ActiveCellLambda,
              typename BcLambda>
    UniformDomain_1DPartitioner(const Neon::Backend&    backend,
                                const ActiveCellLambda& activeCellLambda,
                                const BcLambda&         bcLambda,
                                const int&              blockSize,
                                const Neon::int32_3d&   domainSize,
                                const int&              discreteVoxelSpacing = 1)
    {
        mBlockSize = blockSize;
        mDiscreteVoxelSpacing = discreteVoxelSpacing;

        Neon::int32_3d   block3DSpan(NEON_DIVIDE_UP(domainSize.x, blockSize),
                                     NEON_DIVIDE_UP(domainSize.y, blockSize),
                                     NEON_DIVIDE_UP(domainSize.z, blockSize));
        std::vector<int> nBlockProjectedToZ(block3DSpan.z);

        auto constexpr block3dIdxToBlockOrigin = [&](Neon::int32_3d const& block3dIdx) {
            Neon::int32_3d blockOrigin(block3dIdx.x * blockSize * discreteVoxelSpacing,
                                       block3dIdx.y * blockSize * discreteVoxelSpacing,
                                       block3dIdx.z * blockSize * discreteVoxelSpacing);
            return blockOrigin;
        };

        auto constexpr getVoxelAbsolute3DIdx = [&](Neon::int32_3d const& blockOrigin,
                                                   Neon::int32_3d const& voxelRelative3DIdx) {
            const Neon::int32_3d id(blockOrigin.x + voxelRelative3DIdx.x * discreteVoxelSpacing,
                                    blockOrigin.y + voxelRelative3DIdx.y * discreteVoxelSpacing,
                                    blockOrigin.z + voxelRelative3DIdx.z * discreteVoxelSpacing);
            return id;
        };

        mSpanPartitioner = SpanDecomposition(
            backend,
            activeCellLambda,
            block3dIdxToBlockOrigin,
            getVoxelAbsolute3DIdx,
            block3DSpan,
            blockSize,
            domainSize,
            discreteVoxelSpacing);

        mSpanClassifier = SpanClassifier(
            backend,
            activeCellLambda,
            bcLambda,
            block3dIdxToBlockOrigin,
            getVoxelAbsolute3DIdx,
            block3DSpan,
            blockSize,
            domainSize,
            discreteVoxelSpacing,
            mSpanPartitioner);

        mPartitionSpan = SpanLayout(
            backend,
            mSpanPartitioner,
            mSpanClassifier);
    }

   private:
    int mBlockSize = 0;
    int mDiscreteVoxelSpacing = 0;

    SpanDecomposition mSpanPartitioner;
    SpanClassifier    mSpanClassifier;
    SpanLayout        mPartitionSpan;
};

}  // namespace Neon::domain::tools::partitioning