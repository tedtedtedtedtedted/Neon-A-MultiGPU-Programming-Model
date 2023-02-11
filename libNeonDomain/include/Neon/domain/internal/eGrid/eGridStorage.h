#pragma once
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/memSetOptions.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/set/Capture.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBase.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "eField.h"

#include "Neon/domain/tools/IndexSpaceTable.h"
#include "Neon/domain/tools/PartitionTable.h"
#include "Neon/domain/tools/UniformDomain_1DPartitioner.h"

namespace Neon::domain::internal::eGrid {


struct eStorage
{
   public:
    Neon::domain::tool::IndexSpaceTable<ePartitionIndexSpace> mPartitionIndexSpace;
    int                                                       blockSize;
    int                                                       discreteVoxelSpacing;
    Neon::int32_3d                                            block3DSpan;
    Neon::domain::tool::UniformDomain_1DPartitioner           partitioner;
};

}  // namespace Neon::domain::internal::eGrid
