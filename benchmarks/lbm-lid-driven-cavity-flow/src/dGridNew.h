#pragma once

#include "Neon/domain/dGrid.h"
#include "Neon/domain/tools/GridTransformer.h"
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

class GridTransformation
{
   public:
    using FoundationGrid = Neon::domain::dGrid;
    using PartitionIndexSpace = FoundationGrid::PartitionIndexSpace;

    template <typename T, int C>
    struct Partition : public Neon::domain::dGrid::Partition<T, C>
    {
        using Cell = typename Neon::domain::dGrid::Partition<T, C>::Cell;
        using nghIdx_t = typename Neon::domain::dGrid::Partition<T, C>::nghIdx_t;

        NEON_CUDA_HOST_DEVICE inline auto nghWrite(const Cell& eId,
                                                   nghIdx_t    nghOffset,
                                                   int         card,
                                                   const T&    val) const -> bool
        {
            Cell       cellNgh;
            const bool isValidNeighbour = nghIdx(eId, nghOffset, cellNgh);
            if (isValidNeighbour) {
                operator()(cellNgh, card) = val;
            }
            return isValidNeighbour;
        }
    };

    template <typename T, int C>
    static auto initFieldPartition(const FoundationGrid::Field<T, C>&                   oldField,
                                   Neon::domain::tool::PartitionTable<Partition<T, C>>& partitionTable) -> void
    {
        partitionTable.forEachConfiguration([&](const Neon::Execution& ex,
                                                const Neon::DataView&  dw,
                                                const int              setIdx,
                                                NEON_OUT Partition<T, C>& newPartition) {
            const typename FoundationGrid::Field<T, C>::Partition& oldPartition = oldField.getPartition(ex, dw, setIdx);
            newPartition = Partition<T, C>(oldPartition);
        });
    }

    static auto initPartitionIndexSpace(const FoundationGrid&                                     oldGrid,
                                        Neon::domain::tool::IndexSpaceTable<PartitionIndexSpace>& indexSpaceTable) -> void
    {
        indexSpaceTable.forEachConfiguration([&](const Neon::DataView&         dw,
                                                 const int                     setIdx,
                                                 NEON_OUT PartitionIndexSpace& newIndex) {
            const typename FoundationGrid::PartitionIndexSpace& oldIndexSpace = oldGrid.getPartitionIndexSpace(Neon::DeviceType::CPU,
                                                                                                               setIdx,
                                                                                                               dw);
            newIndex = oldIndexSpace;
        });
    }
};

using dGridNew = Neon::domain::tool::GridTransformer<GridTransformation>::Grid;