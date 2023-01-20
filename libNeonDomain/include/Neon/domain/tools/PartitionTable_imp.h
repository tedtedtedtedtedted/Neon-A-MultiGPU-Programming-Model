#pragma once

#include "Neon/domain/tools/PartitionTable.h"

namespace Neon::domain::tool {

template <typename Partition>
PartitionTable<Partition>::PartitionTable(Neon::Backend& bk)
{  // Setting up the mask for supported executions (i.e host and device | host only | device only)
    for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            mPartitions[Neon::ExecutionUtils::toInt(execution)]
                       [Neon::DataViewUtil::toInt(dw)] =
                           bk.devSet().template newDataSet<Partition>();
        }
    }
    mSetSize = bk.devSet().getCardianlity();
}

template <typename Partition>
auto PartitionTable<Partition>::
    getPartition(Neon::Execution execution,
                 Neon::DataView  dw,
                 Neon::SetIdx    setIdx)
        -> Partition&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const executionInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mPartitions[executionInt][dwInt][setIdx.idx()];
    return output;
}

template <typename Partition>
auto PartitionTable<Partition>::
    getPartition(Neon::Execution execution,
                 Neon::DataView  dw,
                 Neon::SetIdx    setIdx)
        const -> const Partition&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const executionInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mPartitions[executionInt][dwInt][setIdx.idx()];
    return output;
}

template <typename Partition>
template <class Lambda>
auto PartitionTable<Partition>::forEachConfiguration(Lambda& lambda)
{
    for (auto dw : Neon::DataViewUtil::validOptions()) {
        for (auto setIdx = 0; setIdx < mSetSize; setIdx++) {
            lambda(dw, setIdx, getPartition(dw, setIdx));
        }
    }
}

}  // namespace Neon::domain::tool