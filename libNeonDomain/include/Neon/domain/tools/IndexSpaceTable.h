#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
namespace Neon::domain::tool {

/**
 * A helper class to storage and access IndexSpaces parametrically w.r.t Neon::DataView and Neon::Executions
 */
template <typename IndexSpace>
struct IndexSpaceTable
{
    IndexSpaceTable() = default;

    explicit IndexSpaceTable(Neon::Backend& bk);

    auto getIndexSpace(Neon::DataView dw,
                       Neon::SetIdx   setIdx)
        -> IndexSpace&;

    auto getIndexSpace(Neon::DataView dw,
                       Neon::SetIdx   setIdx)
        const -> const IndexSpace&;

    template <class Lambda>
    auto forEachConfiguration(const Lambda& lambda)-> void;

   private:
    using IndexSpacesByDevice = Neon::set::DataSet<IndexSpace>;
    using IndexSpaceByDeviceByDataView = std::array<IndexSpacesByDevice, Neon::DataViewUtil::nConfig>;

    IndexSpaceByDeviceByDataView mIndexSpaceTable;
    int                          mSetSize = 0;
};

}  // namespace Neon::domain::tool
