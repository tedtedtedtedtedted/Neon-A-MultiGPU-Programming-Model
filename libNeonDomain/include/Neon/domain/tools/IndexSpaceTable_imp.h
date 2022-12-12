#pragma once

#include "Neon/domain/tools/IndexSpaceTable.h"

namespace Neon::domain::tool {

template <typename IndexSpace>
IndexSpaceTable<IndexSpace>::IndexSpaceTable(Neon::Backend& bk)
{  // Setting up the mask for supported executions (i.e host and device | host only | device only)
    for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            mIndexSpaceTable[Neon::DataViewUtil::toInt(dw)] =
                bk.devSet().template newDataSet<IndexSpace>();
        }
    }
    mSetSize = bk.devSet().getCardianlity();
}

template <typename IndexSpace>
auto IndexSpaceTable<IndexSpace>::
    getIndexSpace(Neon::DataView dw,
                  Neon::SetIdx   setIdx)
        -> IndexSpace&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    auto&     output = mIndexSpaceTable[dwInt][setIdx.idx()];
    return output;
}

template <typename IndexSpace>
auto IndexSpaceTable<IndexSpace>::
    getIndexSpace(Neon::DataView dw,
                  Neon::SetIdx   setIdx)
        const -> const IndexSpace&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    auto&     output = mIndexSpaceTable[dwInt][setIdx.idx()];
    return output;
}

template <typename IndexSpace>
template <class Lambda>
auto IndexSpaceTable<IndexSpace>::forEachConfiguration(const Lambda& lambda)
    -> void
{
    for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            for (auto setIdx = 0; setIdx < mSetSize; setIdx++) {
                lambda(execution, dw, setIdx, getIndexSpace(execution, dw, setIdx));
            }
        }
    }

}  // namespace Neon::domain::tool