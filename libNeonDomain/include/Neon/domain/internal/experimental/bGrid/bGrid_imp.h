#include "Neon/domain/internal/experimental/bGrid/bGrid.h"

namespace Neon::domain::internal::experimental::bGrid {

struct ZSliceRanges
{
    int32_t zFirst;
    int32_t zLast;
    int64_t nBlocks;
};

template <typename ActiveCellLambda,
          typename BcLambda>
bGrid::bGrid(
    const Neon::Backend&         backend,
    const Neon::int32_3d&        domainSize,
    const ActiveCellLambda&      activeCellLambda,
    const Neon::domain::Stencil& stencil,
    const double_3d&             spacingData,
    const double_3d&             origin,
    const BcLambda&              bcLambda)
    : bGrid(backend, domainSize, activeCellLambda, stencil, 8, 1, spacingData, origin, bcLambda)
{
}


template <typename ActiveCellLambda,
          typename BcLambda>
bGrid::bGrid(const Neon::Backend&         backend,
             const Neon::int32_3d&        domainSize,
             const ActiveCellLambda&      activeCellLambda,
             const Neon::domain::Stencil& stencil,
             const int                    blockSize,
             const int                    discreteVoxelSpacing,
             const double_3d&             spacingData,
             const double_3d&             origin,
             const BcLambda&              bcLambda)
{

    mData = std::make_shared<Data>();
    mData->blockSize = blockSize;
    mData->discreteVoxelSpacing = discreteVoxelSpacing;


    Neon::int32_3d block3DSpan(NEON_DIVIDE_UP(domainSize.x, blockSize),
                               NEON_DIVIDE_UP(domainSize.y, blockSize),
                               NEON_DIVIDE_UP(domainSize.z, blockSize));

    std::vector<int> nBlockProjectedToZ(block3DSpan.z);

    auto constexpr block3dIdxToBlockOrigin = [&](Neon::int32_3d const& block3dIdx) {
        Neon::int32_3d blockOrigin(block3dIdx.x * blockSize * discreteVoxelSpacing,
                                   block3dIdx.y * blockSize * discreteVoxelSpacing,
                                   block3dIdx.z * blockSize * discreteVoxelSpacing);
        return blockOrigin;
    };

    auto constexpr getVoxelAbsolute3DIdx = [&](Neon::int32_3d const& blockOrigin,
                                               Neon::int32_3d const& voxelRelative3DIdx) {
        const Neon::int32_3d id(blockOrigin.x + voxelRelative3DIdx.x * discreteVoxelSpacing,
                                blockOrigin.y + voxelRelative3DIdx.y * discreteVoxelSpacing,
                                blockOrigin.z + voxelRelative3DIdx.z * discreteVoxelSpacing);
        return id;
    };

    mData->mSpanPartitioner = details::SpanDecomposition(
        backend,
        activeCellLambda,
        block3dIdxToBlockOrigin,
        getVoxelAbsolute3DIdx,
        block3DSpan,
        blockSize,
        domainSize,
        discreteVoxelSpacing);

    mData->mSpanClassifier = details::SpanClassifier(
        backend,
        activeCellLambda,
        bcLambda,
        block3dIdxToBlockOrigin,
        getVoxelAbsolute3DIdx,
        block3DSpan,
        blockSize,
        domainSize,
        discreteVoxelSpacing,
        mData->mSpanPartitioner);

    mData->mPartitionSpan = details::SpanLayout(
        backend,
        mData->mSpanPartitioner,
        mData->mSpanClassifier);

    Neon::MemoryOptions memOptionsAoS(
        Neon::DeviceType::CPU,
        Neon::Allocator::MALLOC,
        Neon::DeviceType::CUDA,
        ((backend.devType() == Neon::DeviceType::CUDA)
             ? Neon::Allocator::CUDA_MEM_DEVICE
             : Neon::Allocator::NULL_MEM),
        Neon::MemoryLayout::arrayOfStructs);

    {  // Multi-XPU vector of Block origins (O.x,O.y,O.z)
        mData->mOrigins = mData->mPartitionSpan.allocateBlockOriginMemSet(
            backend,
            Neon::Backend::mainStreamIdx);
    }

    {  // Stencil linear/relative index
        mData->mStencilNghIndex =
            mData->mPartitionSpan.allocateStencilRelativeIndexMap(
                backend,
                Neon::Backend::mainStreamIdx,
                stencil);
    }

    {  // Allocating (mActiveMask) the block bitmask that identify active voxels.

        mData->mActiveMask = mData->mPartitionSpan.allocateActiveMaskMemSet(backend,
                                                                            Neon::Backend::mainStreamIdx,
                                                                            activeCellLambda,
                                                                            domainSize,
                                                                            blockSize,
                                                                            discreteVoxelSpacing);
    }


    {  // Neighbor blocks
        mData->mNeighbourBlocks = mData->mPartitionSpan.allocateBlockConnectivityMemSet(
            backend,
            Neon::Backend::mainStreamIdx);
    }
}

template <typename T, int C>
auto bGrid::newField(const std::string          name,
                     int                        cardinality,
                     T                          inactiveValue,
                     Neon::DataUse              dataUse,
                     const Neon::MemoryOptions& memoryOptions) const -> Field<T, C>
{
    bField<T, C> field(name, *this, cardinality, inactiveValue, dataUse, memoryOptions, Neon::domain::haloStatus_et::ON);

    return field;
}


template <typename LoadingLambda>
auto bGrid::getContainer(const std::string& name,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda) const -> Neon::set::Container
{
    Neon::set::Container kContainer = Neon::set::Container::factory(name,
                                                                    Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                    *this,
                                                                    lambda,
                                                                    blockSize,
                                                                    [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}

template <typename LoadingLambda>
auto bGrid::getContainer(const std::string& name,
                         LoadingLambda      lambda) const -> Neon::set::Container
{
    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     defaultBlockSize,
                                                                     [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}

template <typename T>
auto bGrid::newPatternScalar() const -> Neon::template PatternScalar<T>
{
//    // TODO this sets the numBlocks for only Standard dataView.
//    auto pattern = Neon::PatternScalar<T>(getBackend(), Neon::sys::patterns::Engine::CUB);
//    for (SetIdx id = 0; id < mData->mNumBlocks.cardinality(); id++) {
//        pattern.getBlasSet(Neon::DataView::STANDARD).getBlas(id.idx()).setNumBlocks(uint32_t(mData->mNumBlocks[id]));
//    }
//    return pattern;
    NEON_DEV_UNDER_CONSTRUCTION("");
}


template <typename T>
auto bGrid::dot(const std::string&               name,
                Field<T>&                        input1,
                Field<T>&                        input2,
                Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    return Neon::set::Container::factoryOldManaged(
        name,
        Neon::set::internal::ContainerAPI::DataViewSupport::on,
        Neon::set::ContainerPatternType::reduction,
        *this, [&](Neon::set::Loader& loader) {
            loader.load(input1);
            if (input1.getUid() != input2.getUid()) {
                loader.load(input2);
            }

            return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                if (dataView != Neon::DataView::STANDARD) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation on bGrid works only on standard dataview";
                    exc << "Input dataview is" << Neon::DataViewUtil::toString(dataView);
                    NEON_THROW(exc);
                }

                if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                    NEON_THROW(exc);
                }

                if (getBackend().devType() == Neon::DeviceType::CUDA) {
                    scalar.setStream(streamIdx, dataView);

                    // calc dot product and store results on device
                    input1.dot(scalar.getBlasSet(dataView),
                               input2,
                               scalar.getTempMemory(dataView, Neon::DeviceType::CUDA),
                               dataView);

                    // move to results to host
                    scalar.getTempMemory(dataView,
                                         Neon::DeviceType::CPU)
                        .template updateFrom<Neon::run_et::et::async>(
                            scalar.getBlasSet(dataView).getStream(),
                            scalar.getTempMemory(dataView, Neon::DeviceType::CUDA));

                    // sync
                    scalar.getBlasSet(dataView).getStream().sync();

                    // read the results
                    scalar() = scalar.getTempMemory(dataView, Neon::DeviceType::CPU).elRef(0, 0, 0);
                } else {

                    scalar() = 0;
                    input1.forEachActiveCell(
                        [&](const Neon::index_3d& idx,
                            const int&            cardinality,
                            T&                    in1) {
                            scalar() += in1 * input2(idx, cardinality);
                        },
                        Neon::computeMode_t::computeMode_e::seq);
                }
            };
        });
}

template <typename T>
auto bGrid::norm2(const std::string&               name,
                  Field<T>&                        input,
                  Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    return Neon::set::Container::factoryOldManaged(
        name,
        Neon::set::internal::ContainerAPI::DataViewSupport::on,
        Neon::set::ContainerPatternType::reduction,
        *this, [&](Neon::set::Loader& loader) {
            loader.load(input);


            return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                if (dataView != Neon::DataView::STANDARD) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation on bGrid works only on standard dataview";
                    exc << "Input dataview is" << Neon::DataViewUtil::toString(dataView);
                    NEON_THROW(exc);
                }

                if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                    NeonException exc("bGrid");
                    exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                    NEON_THROW(exc);
                }

                if (getBackend().devType() == Neon::DeviceType::CUDA) {
                    scalar.setStream(streamIdx, dataView);

                    // calc dot product and store results on device
                    input.norm2(scalar.getBlasSet(dataView),
                                scalar.getTempMemory(dataView, Neon::DeviceType::CUDA),
                                dataView);

                    // move to results to host
                    scalar.getTempMemory(dataView,
                                         Neon::DeviceType::CPU)
                        .template updateFrom<Neon::run_et::et::async>(
                            scalar.getBlasSet(dataView).getStream(),
                            scalar.getTempMemory(dataView, Neon::DeviceType::CUDA));

                    // sync
                    scalar.getBlasSet(dataView).getStream().sync();

                    // read the results
                    scalar() = scalar.getTempMemory(dataView, Neon::DeviceType::CPU).elRef(0, 0, 0);
                } else {

                    scalar() = 0;
                    input.forEachActiveCell(
                        [&]([[maybe_unused]] const Neon::index_3d& idx,
                            [[maybe_unused]] const int&            cardinality,
                            T&                                     in) {
                            scalar() += in * in;
                        },
                        Neon::computeMode_t::computeMode_e::seq);
                }
                scalar() = std::sqrt(scalar());
            };
        });
}
}  // namespace Neon::domain::internal::experimental::bGrid