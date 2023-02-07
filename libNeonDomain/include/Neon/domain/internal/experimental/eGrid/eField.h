#pragma once
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/internal/experimental/eGrid/ePartition.h"
#include "Neon/set/patterns/BlasSet.h"

namespace Neon::domain::internal::experimental::eGrid::details {
class eGrid;


template <typename T, int C = 0>
class eField : public Neon::domain::interface::FieldBaseTemplate<T,
                                                                 C,
                                                                 eGrid,
                                                                 bPartition<T, C>,
                                                                 int>
{
    friend eGrid;

   public:
    using Type = T;
    using Grid = eGrid;
    using Field = eField;
    using Partition = bPartition<T, C>;
    using Cell = eCell;
    using ngh_idx = typename Partition::nghIdx_t;

    eField(const std::string&             name,
           const eGrid&                   grid,
           int                            cardinality,
           T                              outsideVal,
           Neon::DataUse                  dataUse,
           const Neon::MemoryOptions&     memoryOptions,
           Neon::domain::haloStatus_et::e haloStatus);

    eField() = default;

    virtual ~eField() = default;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView) const -> const Partition&;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView) -> Partition&;

    auto getPartition(Neon::Execution,
                      Neon::SetIdx,
                      const Neon::DataView& dataView) const -> const Partition& final;

    auto getPartition(Neon::Execution,
                      Neon::SetIdx,
                      const Neon::DataView& dataView) -> Partition& final;

    auto isInsideDomain(const Neon::index_3d& idx) const -> bool;


    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const -> T final;

    auto getReference(const Neon::index_3d& idx,
                      const int&            cardinality) -> T& final;


    auto haloUpdate(Neon::set::HuOptions& opt) const -> void final;

    auto haloUpdate(Neon::set::HuOptions& opt) -> void final;

    auto updateIO(int streamId = 0) -> void final;

    auto updateCompute(int streamId = 0) -> void final;

    auto getSharedMemoryBytes(const int32_t stencilRadius) const -> size_t;

    auto getMem() -> Neon::set::MemSet_t<T>&;

    auto dot(Neon::set::patterns::BlasSet<T>& blasSet,
             const eField<T>&                 input,
             Neon::set::MemDevSet<T>&         output,
             const Neon::DataView&            dataView) -> void;

    auto norm2(Neon::set::patterns::BlasSet<T>& blasSet,
               Neon::set::MemDevSet<T>&         output,
               const Neon::DataView&            dataView) -> void;


    auto forEachActiveCell(const std::function<void(const Neon::index_3d&,
                                                    const int& cardinality,
                                                    T&)>&     fun,
                           Neon::computeMode_t::computeMode_e mode = Neon::computeMode_t::computeMode_e::par) -> void override;


   private:
    auto getRef(const Neon::index_3d& idx, const int& cardinality) const -> T&;


    enum PartitionBackend
    {
        cpu = 0,
        gpu = 1,
    };

    struct Data
    {

        std::shared_ptr<eGrid> grid;

        Neon::set::MemSet_t<T> mem;

        int mCardinality;

        std::array<
            std::array<
                Neon::set::DataSet<Partition>,
                Neon::DataViewUtil::nConfig>,
            2>  // 2 for host and device
            partitions;
    };
    std::shared_ptr<Data> mData;
};
}  // namespace Neon::domain::internal::experimental::eGrid::details

#include "Neon/domain/internal/eGrid/eField_imp.h"