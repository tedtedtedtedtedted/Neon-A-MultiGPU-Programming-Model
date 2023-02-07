#include "Neon/domain/internal/experimental/bGrid/SpanLayout.h"
#include "Neon/core/core.h"
#include "Neon/domain/internal/experimental/bGrid/bCell.h"

namespace Neon::domain::internal::experimental::bGrid::details {

SpanLayout::SpanLayout(Neon::Backend const&     backend,
                       SpanDecomposition const& spanPartitioner,
                       SpanClassifier const&    spanClassifier)
{
    mMemOptionsAoS = Neon::MemoryOptions(
        Neon::DeviceType::CPU,
        Neon::Allocator::MALLOC,
        Neon::DeviceType::CUDA,
        backend.devType() == Neon::DeviceType::CUDA
            ? Neon::Allocator::CUDA_MEM_DEVICE
            : Neon::Allocator::NULL_MEM,
        Neon::MemoryLayout::arrayOfStructs);


    mSpanPartitioner = &spanPartitioner;
    mSpanClassifierPtr = &spanClassifier;

    mCountXpu = backend.devSet().setCardinality();
    mDataByPartition = backend.devSet().newDataSet<InfoByPartition>();
    // Setting up internal and boudary indexes
    mDataByPartition.forEachSetIdx([&](Neon::SetIdx const& setIdx,
                                       InfoByPartition&    data) {
        int counter = 0;
        for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
            int toAdd = spanClassifier.countInternal(setIdx, byDomain);
            data.getInternal()(byDomain).first = counter;
            data.getInternal()(byDomain).count = toAdd;
            counter += toAdd;
        }

        for (auto const& byDirection : {ByDirection::up, ByDirection::down}) {
            for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
                int   toAdd = spanClassifier.countBoundary(setIdx, byDirection, byDomain);
                auto& info = data.getBoundary(byDirection)(byDomain);
                info.first = counter;
                info.count = toAdd;
                counter += toAdd;
            }
        }
    });


    mDataByPartition.forEachSetIdx([&](Neon::SetIdx const& setIdx,
                                       InfoByPartition&    data) -> void {
        for (auto const& byDirection : {ByDirection::up, ByDirection::down}) {
            for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {

                auto const& ghostTarget = getTargetGhost(setIdx, byDirection);

                data.getGhost(byDirection).getGhost() = ghostTarget;
                auto& info = data.getGhost(byDirection)(byDomain);

                auto& ghostData = mDataByPartition[ghostTarget.setIdx];
                auto& ghostInfo = ghostData.getBoundary(ghostTarget.byDirection)(byDomain);

                info.first = ghostInfo.first;
                info.count = ghostInfo.count;
            }
        }
    });
}

auto SpanLayout::getBoundsInternal(
    SetIdx setIdx)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getInternal().operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getInternal().operator()(byDomain).count;
    }
    return result;
}

auto SpanLayout::getBoundsInternal(
    SetIdx   setIdx,
    ByDomain byDomain)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getInternal().operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getInternal().operator()(byDomain).count;
    return result;
}

auto SpanLayout::getBoundsBoundary(
    SetIdx      setIdx,
    ByDirection byDirection)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getBoundary(byDirection).operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).count;
    }
    return result;
}

auto SpanLayout::getBoundsBoundary(
    SetIdx      setIdx,
    ByDirection byDirection,
    ByDomain    byDomain)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).count;
    return result;
}

auto SpanLayout::getGhostBoundary(SetIdx setIdx, ByDirection byDirection)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getGhost(byDirection).operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).count;
    }
    return result;
}

auto SpanLayout::getGhostBoundary(
    SetIdx      setIdx,
    ByDirection byDirection,
    ByDomain    byDomain)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).count;
    return result;
}

auto SpanLayout::getGhostTarget(SetIdx setIdx, ByDirection byDirection)
    const -> SpanLayout::GhostTarget
{
    return mDataByPartition[setIdx].getGhost(byDirection).getGhost();
}

auto SpanLayout::getLocalPointOffset(
    SetIdx          setIdx,
    const int32_3d& point) const -> std::pair<bool, int32_t>
{
    auto findings = findPossiblyLocalPointOffset(setIdx, point);
    if (std::get<0>(findings)) {
        auto classificationOffset = getClassificationOffset(setIdx,
                                                            std::get<2>(findings),
                                                            std::get<3>(findings),
                                                            std::get<4>(findings));
        return {true, std::get<1>(findings) + classificationOffset};
    }
    return {false, -1};
}


auto SpanLayout::findPossiblyLocalPointOffset(
    SetIdx          setIdx,
    const int32_3d& point)
    const -> std::tuple<bool, int32_t, ByPartition, ByDirection, ByDomain>
{
    for (auto byPartition : {ByPartition::internal, ByPartition::boundary}) {
        for (auto byDirection : {ByDirection::up, ByDirection::down}) {
            if (byPartition == ByPartition::internal && byDirection == ByDirection::down) {
                continue;
            }
            for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                auto const& mapper = mSpanClassifierPtr->getMapper3Dto1D(setIdx,
                                                                         byPartition,
                                                                         byDirection,
                                                                         byDomain);
                auto const  infoPtr = mapper.getMetadata(point);
                if (infoPtr == nullptr) {
                    return {false, -1, byPartition, byDirection, byDomain};
                }
                return {true, *infoPtr, byPartition, byDirection, byDomain};
            }
        }
    }
}

auto SpanLayout::getClassificationOffset(
    Neon::SetIdx setIdx,
    ByPartition  byPartition,
    ByDirection  byDirection,
    ByDomain     byDomain)
    const -> int32_t
{
    if (byPartition == ByPartition::internal) {
        return this->getBoundsInternal(setIdx, byDomain).first;
    }
    return this->getBoundsBoundary(setIdx, byDirection, byDomain).first;
}

auto SpanLayout::findNeighbourOfInternalPoint(
    SetIdx                setIdx,
    const Neon::int32_3d& point,
    const Neon::int32_3d& offset)
    const -> std::pair<bool, int32_t>
{
    // Neighbours of internal points can be internal or boundary
    auto nghPoint = point + offset;
    return getLocalPointOffset(setIdx, nghPoint);
}

auto SpanLayout::findNeighbourOfBoundaryPoint(
    SetIdx                setIdx,
    const Neon::int32_3d& point,
    const Neon::int32_3d& nghOffset)
    const -> std::pair<bool, int32_t>
{
    // Neighbours of internal points can be internal or boundary
    auto nghPoint = point + nghOffset;
    auto findings = findPossiblyLocalPointOffset(setIdx, nghPoint);
    if (std::get<0>(findings)) {
        auto classificationOffset = getClassificationOffset(setIdx,
                                                            std::get<2>(findings),
                                                            std::get<3>(findings),
                                                            std::get<4>(findings));
        return {true, std::get<1>(findings) + classificationOffset};
    }
    // We need to search on local partitions
    // We select the target partition based on the .z component of the offset
    int          partitionOffset = nghOffset.z > 0 ? +1 : -1;
    Neon::SetIdx nghSetIdx = (setIdx.idx() + mCountXpu + partitionOffset) % mCountXpu;

    findings = findPossiblyLocalPointOffset(nghSetIdx, nghPoint);
    if (std::get<0>(findings)) {
        // Ghost direction is the opposite w.r.t. the neighbour partition direction
        ByDirection ghostByDirection = nghOffset.z > 0
                                           ? ByDirection::down
                                           : ByDirection::up;
        ByDomain    ghostByDomain = std::get<4>(findings);

        Bounds ghostBounds = getGhostBoundary(setIdx,
                                              ghostByDirection,
                                              ghostByDomain);

        return {true, std::get<1>(findings) + ghostBounds.first};
    }
    return {false, -1};
}

auto SpanLayout::allocateBlockOriginMemSet(
    Neon::Backend const& backend,
    int                  stream)
    const -> Neon::set::MemSet_t<Neon::int32_3d>
{
    // Multi-XPU vector of Block origins (O.x,O.y,O.z)
    auto originsMemSet = backend.devSet().template newMemSet<Neon::int32_3d>(
        Neon::DataUse::IO_COMPUTE,
        1,
        mMemOptionsAoS,
        mSpanPartitioner->getNumBlockPerPartition().newType<uint64_t>());

    backend.devSet().forEachSetIdxSeq(
        [&](Neon::SetIdx const& setIdx) {
            for (auto byPartition : {ByPartition::internal}) {
                const auto byDirection = ByDirection::up;
                for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                    auto const& mapperVec = mSpanClassifierPtr->getMapper1Dto3D(
                        setIdx,
                        byPartition,
                        byDirection,
                        byDomain);

                    auto const start = this->getBoundsInternal(setIdx, byDomain).first;
                    for (uint64_t j = 0; j < mapperVec.size(); j++) {
                        auto const& point3d = mapperVec[j];
                        originsMemSet.eRef(setIdx, j, 0) = point3d;
                    }
                }
            }

            for (auto byPartition : {ByPartition::boundary}) {
                for (auto byDirection : {ByDirection::up, ByDirection::down}) {

                    for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                        auto const& mapperVec = mSpanClassifierPtr->getMapper1Dto3D(
                            setIdx,
                            byPartition,
                            byDirection,
                            byDomain);

                        auto const start = this->getBoundsBoundary(setIdx, byDirection, byDomain).first;
                        for (uint64_t j = 0; j < mapperVec.size(); j++) {
                            auto const& point3d = mapperVec[j];
                            originsMemSet.eRef(setIdx, j, 0) = point3d;
                        }
                    }
                }
            }
        });

    originsMemSet.updateCompute(backend, stream);

    return originsMemSet;
}

auto SpanLayout::allocateStencilRelativeIndexMap(
    const Backend&               backend,
    int                          stream,
    const Neon::domain::Stencil& stencil) const -> Neon::set::MemSet_t<int8_3d>
{

    auto stencilNghSize = backend.devSet().template newDataSet<uint64_t>(
        stencil.neighbours().size());

    Neon::set::MemSet_t<int8_3d> stencilNghIndex = backend.devSet().template newMemSet<int8_3d>(
        Neon::DataUse::IO_COMPUTE,
        1,
        mMemOptionsAoS,
        stencilNghSize);

    for (int32_t c = 0; c < stencilNghIndex.cardinality(); ++c) {
        SetIdx devID(c);
        for (int64_t s = 0; s < int64_t(stencil.neighbours().size()); ++s) {
            stencilNghIndex.eRef(c, s).x = static_cast<int8_3d::Integer>(stencil.neighbours()[s].x);
            stencilNghIndex.eRef(c, s).y = static_cast<int8_3d::Integer>(stencil.neighbours()[s].y);
            stencilNghIndex.eRef(c, s).z = static_cast<int8_3d::Integer>(stencil.neighbours()[s].z);
        }
    }

    stencilNghIndex.updateCompute(backend, stream);
}

auto SpanLayout::allocateBlockConnectivityMemSet(
    const Backend& backend,
    int            stream) const -> Neon::set::MemSet_t<uint32_t>
{
    auto numBlocks =
        mSpanPartitioner->getNumBlockPerPartition().newType<uint64_t>();

    auto neighbourBlocks = backend.devSet().template newMemSet<uint32_t>(
        Neon::DataUse::IO_COMPUTE,
        3 * 3 * 3 - 1,
        mMemOptionsAoS,
        mSpanPartitioner->getNumBlockPerPartition().newType<uint64_t>());

    backend.devSet().forEachSetIdxSeq(
        [&](Neon::SetIdx const& setIdx) {
            for (auto byPartition : {ByPartition::internal}) {
                const auto byDirection = ByDirection::up;
                for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                    auto const& mapperVec = mSpanClassifierPtr->getMapper1Dto3D(
                        setIdx,
                        byPartition,
                        byDirection,
                        byDomain);

                    auto const start = this->getBoundsInternal(setIdx, byDomain).first;
                    for (uint64_t blockIdx = 0; blockIdx < mapperVec.size(); blockIdx++) {
                        auto const& point3d = mapperVec[blockIdx];
                        for (int16_t k = -1; k < 2; k++) {
                            for (int16_t j = -1; j < 2; j++) {
                                for (int16_t i = -1; i < 2; i++) {
                                    if (i == 0 && j == 0 && k == 0) {

                                        Neon::int16_3d const offset(i, j, k);

                                        auto findings = findNeighbourOfInternalPoint(
                                            setIdx,
                                            point3d, offset.newType<int32_t>());

                                        uint32_t const noNeighbour = std::numeric_limits<uint32_t>::max();
                                        uint32_t       targetNgh = noNeighbour;
                                        if (findings.first) {
                                            targetNgh = findings.second;
                                        }
                                        neighbourBlocks.eRef(setIdx,
                                                             blockIdx,
                                                             bCell::getNeighbourBlockID(offset)) = targetNgh;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (auto byPartition : {ByPartition::boundary}) {
                for (auto byDirection : {ByDirection::up, ByDirection::down}) {

                    for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                        auto const& mapperVec = mSpanClassifierPtr->getMapper1Dto3D(
                            setIdx,
                            byPartition,
                            byDirection,
                            byDomain);

                        auto const start = this->getBoundsBoundary(setIdx, byDirection, byDomain).first;
                        for (int64_t blockIdx = 0; blockIdx < int64_t(mapperVec.size()); blockIdx++) {
                            auto const& point3d = mapperVec[blockIdx];
                            for (int16_t k = -1; k < 2; k++) {
                                for (int16_t j = -1; j < 2; j++) {
                                    for (int16_t i = -1; i < 2; i++) {
                                        if (i == 0 && j == 0 && k == 0) {

                                            Neon::int16_3d const offset(i, j, k);

                                            auto findings = findNeighbourOfBoundaryPoint(
                                                setIdx,
                                                point3d,
                                                offset.newType<int32_t>());

                                            uint32_t const noNeighbour = std::numeric_limits<uint32_t>::max();
                                            uint32_t       targetNgh = noNeighbour;
                                            if (findings.first) {
                                                targetNgh = findings.second;
                                            }
                                            neighbourBlocks.eRef(setIdx,
                                                                 blockIdx,
                                                                 bCell::getNeighbourBlockID(offset)) = targetNgh;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

    neighbourBlocks.updateCompute(backend, stream);

    return neighbourBlocks;
}


}  // namespace Neon::domain::internal::experimental::bGrid::details
