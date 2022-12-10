#pragma once

#include "Neon/domain/tools/PartitionStorage.h"

namespace Neon::domain::tool {

template <typename Partition>
PartitionStorage<Partition>::PartitionStorage(Neon::Backend& bk)
{  // Setting up the mask for supported executions (i.e host and device | host only | device only)
    for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            mPartitions[Neon::ExecutionUtils::toInt(execution)]
                       [Neon::DataViewUtil::toInt(dw)] =
                           bk.devSet().newDataSet<Partition>();
        }
    }
}

template <typename Partition>
auto PartitionStorage<Partition>::
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
auto PartitionStorage<Partition>::
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

}  // namespace Neon::domain::tool