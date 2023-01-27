#include "Neon/domain/internal/experimental/bGrid/SpanPartitioner.h"

namespace Neon::domain::internal::experimental::bGrid {

auto SpanPartitioner::getNumBlockPerPartition() const -> const Neon::set::DataSet<int64_t>&
{
    return mNumBlocks;
}

auto SpanPartitioner::getFirstZSliceIdx() const -> const Neon::set::DataSet<int32_t>&
{
    return mZFirstIdx;
}

auto SpanPartitioner::getLastZSliceIdx() const -> const Neon::set::DataSet<int32_t>&
{
    return mZLastIdx;
}

}  // namespace Neon::domain::internal::experimental::bGrid