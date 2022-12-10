#pragma once

#include "Neon/set/Containter.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/patterns/PatternScalar.h"

/**
 * template <typename FoundationGrid>
 * GridTransformation {
 *      using PartitionIndexSpace
 *      using Partition
 *      using FoundationGrid
 *
 *      getLaunchParameters(Neon::DataView        dataView,
const Neon::index_3d& blockSize,
const size_t&         shareMem)
 * }
 */
namespace Neon::domain::tool::details {

template <typename T,
          int card,
          typename GridTransformation>
class tField;

template <typename GridTransformation>
class tGrid : public Neon::domain::interface::GridBaseTemplate<tGrid<GridTransformation>,
                                                               typename GridTransformation::Partition::Cell>
{
   public:
    template <class T, int card>
    using Field = tField<T, card, GridTransformation>;
    template <class T, int card>
    using Partition = typename GridTransformation::template Partition<T, card>;
    using PartitionIndexSpace = typename GridTransformation::PartitionIndexSpace;

   private:
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<tGrid<GridTransformation>,
                                                                       typename GridTransformation::Partition::Cell>;

    using FoundationGrid = typename GridTransformation::FoundationGrid;

   public:
    explicit tGrid(FoundationGrid& foundationGrid);

    tGrid();
    virtual ~tGrid();
    tGrid(const tGrid& other);                 // copy constructor
    tGrid(tGrid&& other) noexcept;             // move constructor
    tGrid& operator=(const tGrid& other);      // copy assignment
    tGrid& operator=(tGrid&& other) noexcept;  // move assignment

    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         shareMem) const
        -> Neon::set::LaunchParameters;

    auto getPartitionIndexSpace(Neon::DeviceType devE,
                                SetIdx           setIdx,
                                Neon::DataView   dataView)
        -> const PartitionIndexSpace&;

    template <typename T, int C = 0>
    auto newField(const std::string   fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::IO_COMPUTE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const
        -> Neon::set::Container;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView)
        -> Neon::set::KernelConfig;

    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    auto getProperties(const Neon::index_3d& idx) const
        -> typename GridBaseTemplate::CellProperties final;

   private:
    struct Storage
    {
        using IndexSpaceInformation = std::array<Neon::set::DataSet<PartitionIndexSpace>, Neon::DataViewUtil::nConfig>;

        FoundationGrid        foundationGrid;
        IndexSpaceInformation indexSpace;
        // std::array<Neon::set::DataSet<PartitionIndexSpace>> partitionIndexSpaceVec;
    };

    std::shared_ptr<Storage> mStorage;
};

}  // namespace Neon::domain::tool::details