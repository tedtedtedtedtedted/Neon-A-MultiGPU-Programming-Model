#include "Neon/domain/internal/experimental/bGrid/SpanDecomposition.h"

namespace Neon::domain::internal::experimental::bGrid::details {

auto SpanDecomposition::getNumBlockPerPartition() const -> const Neon::set::DataSet<int64_t>&
{
    return mNumBlocks;
}

auto SpanDecomposition::getFirstZSliceIdx() const -> const Neon::set::DataSet<int32_t>&
{
    return mZFirstIdx;
}

auto SpanDecomposition::getLastZSliceIdx() const -> const Neon::set::DataSet<int32_t>&
{
    return mZLastIdx;
}

}  // namespace Neon::domain::internal::experimental::bGrid