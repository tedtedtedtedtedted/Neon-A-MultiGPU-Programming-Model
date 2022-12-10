#pragma once
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/tools/PartitionStorage.h"

namespace Neon::domain::tool::details {

template <typename GridTransformation>
class tGrid;


template <typename T, int C, typename GridTransformation>
class tField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 tGrid<GridTransformation>,
                                                                 typename GridTransformation::template dPartition<T, C>,
                                                                 int>
{
    friend tGrid<GridTransformation>;

   public:
    static constexpr int Cardinality = C;
    using Type = T;
    using Self = tField<Type, Cardinality, GridTransformation>;
    using Grid = tGrid<GridTransformation>;
    using Field = tField<Type, Cardinality, GridTransformation>;
    using Partition = typename GridTransformation::template dPartition<T, C>;
    using Cell = typename Partition::Cell;
    using ngh_idx = typename Partition::nghIdx_t;  // for compatibility with eGrid

   private:
    using FoundationGrid = typename GridTransformation::FoundationGrid;
    using FoundationField = typename GridTransformation::FoundationGrid::template Field<T, C>;

   public:
    tField() = default;
    ~tField() = default;


    /**
     * Returns the metadata associated with the element in location idx.
     * If the element is not active (it does not belong to the voxelized domain),
     * then the default outside value is returned.
     */
    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const
        -> Type final;

    auto haloUpdate(Neon::set::HuOptions& opt) const
        -> void final;

    auto haloUpdate(SetIdx setIdx, Neon::set::HuOptions& opt) const
        -> void;  // TODO add this function to the API if benchmarks boost is reasonable -> void final;

    auto haloUpdate(Neon::set::HuOptions& opt)
        -> void final;

    auto haloUpdate(SetIdx setIdx, Neon::set::HuOptions& opt)
        -> void;  // TODO add this function to the API if benchmarks boost is reasonable -> void final;

    virtual auto getReference(const Neon::index_3d& idx,
                              const int&            cardinality)
        -> Type& final;

    auto updateCompute(int streamSetId)
        -> void;

    auto updateIO(int streamSetId)
        -> void;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView = Neon::DataView::STANDARD)
        const
        -> const Partition&;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView = Neon::DataView::STANDARD)
        -> Partition&;

    /**
     * Return a constant reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Partition& final;
    /**
     * Return a reference to a specific partition based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Partition& final;

    static auto swap(Field& A, Field& B) -> void;

   private:

    auto updateCompute(const Neon::set::StreamSet& streamSet) -> void;


    auto updateIO(const Neon::set::StreamSet& streamSet)
        -> void;


    auto getLaunchInfo(const Neon::DataView dataView) const -> Neon::set::LaunchParameters;


    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdate(const Neon::Backend& bk,
                    bool                 startWithBarrier = true,
                    int                  streamSetIdx = 0)
        -> void;

    template <Neon::set::TransferMode transferMode_ta>
    auto haloUpdate(const Neon::Backend& bk,
                    int                  cardinality,
                    bool                 startWithBarrier = true,
                    int                  streamSetIdx = 0)
        -> void;


    tField(const std::string&                        fieldUserName,
           Neon::DataUse                             dataUse,
           const Neon::MemoryOptions&                memoryOptions,
           const Grid&                               grid,
           const Neon::set::DataSet<Neon::index_3d>& dims,
           int                                       cardinality){

    }

    struct Storage
    {
        FoundationField                                 foundationField;
        Neon::domain::tool::PartitionStorage<Partition> partitions;
    };
};

}  // namespace Neon::domain::tool::details