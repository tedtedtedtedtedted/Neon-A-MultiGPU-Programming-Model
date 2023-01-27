#pragma once
#include "Neon/core/core.h"

#include "Neon/set/Containter.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"

#include "Neon/domain/internal/experimental/bGrid/bCell.h"
#include "Neon/domain/internal/experimental/bGrid/bField.h"
#include "Neon/domain/internal/experimental/bGrid/bPartition.h"
#include "Neon/domain/internal/experimental/bGrid/bPartitionIndexSpace.h"

#include "Neon/domain/patterns/PatternScalar.h"

#include "Neon/domain/tools/IndexSpaceTable.h"
#include "Neon/domain/tools/PointHashTable.h"

namespace Neon::domain::internal::experimental::bGrid {

template <typename T, int C>
class bField;

class Classifier
{
    Classifier() = default;
    Classifier(const Neon::Backend&);

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

}  // namespace Neon::domain::internal::experimental::bGrid
