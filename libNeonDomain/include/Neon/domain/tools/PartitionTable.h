#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
namespace Neon::domain::tool {

/**
 * A helper class to storage and access partitions parametrically w.r.t Neon::DataView and Neon::Executions
 */
template <typename Partition>
struct PartitionTable
{
    PartitionTable() = default;

    explicit PartitionTable(Neon::Backend& bk);

    auto getPartition(Neon::Execution execution,
                      Neon::DataView  dw,
                      Neon::SetIdx    setIdx)
        -> Partition&;

    auto getPartition(Neon::Execution execution,
                      Neon::DataView  dw,
                      Neon::SetIdx    setIdx)
        const -> const Partition&;

    template <class Lambda>
    auto forEachConfiguration(Lambda& lambda);

   private:
    using PartitionsByDevice = Neon::set::DataSet<Partition>;
    using PartitionByDeviceByDataView = std::array<PartitionsByDevice, Neon::DataViewUtil::nConfig>;
    using PartitionByDeviceByDataViewByExecution = std::array<PartitionByDeviceByDataView, Neon::ExecutionUtils::numConfigurations>;

    PartitionByDeviceByDataViewByExecution mPartitions;
    int                                    mSetSize = 0;
};

}  // namespace Neon::domain::tool
