#pragma once
#include "Neon/core/core.h"

#include "Neon/set/Containter.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"

#include "Neon/domain/internal/experimental/bGrid/SpanDecomposition.h"

#include "Neon/domain/patterns/PatternScalar.h"

#include "Neon/domain/tools/IndexSpaceTable.h"
#include "Neon/domain/tools/PointHashTable.h"

namespace Neon::domain::tools::partitioning {

enum struct ByPartition
{
    internal = 0,
    boundary = 1
};

enum struct ByDomain
{
    bc = 0,
    bulk = 1
};

enum struct ByDirection
{
    up = 0,
    down = 1
};

}  // namespace Neon::domain::tools::partitioning
