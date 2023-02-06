#include "Neon/domain/internal/experimental/bGrid/PartitionSpan.h"
#include "Neon/core/core.h"

namespace Neon::domain::internal::experimental::bGrid::details {

PartitionSpan::PartitionSpan(Neon::Backend const&   backend,
                             SpanPartitioner const& spanPartitioner,
                             SpanClassifier const&  spanClassifier)
{
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

auto PartitionSpan::getBoundsInternal(
    SetIdx setIdx)
    const -> PartitionSpan::Bounds
{
    PartitionSpan::Bounds result{};
    result.first = mDataByPartition[setIdx].getInternal().operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getInternal().operator()(byDomain).count;
    }
    return result;
}

auto PartitionSpan::getBoundsInternal(
    SetIdx   setIdx,
    ByDomain byDomain)
    const -> PartitionSpan::Bounds
{
    PartitionSpan::Bounds result{};
    result.first = mDataByPartition[setIdx].getInternal().operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getInternal().operator()(byDomain).count;
    return result;
}

auto PartitionSpan::getBoundsBoundary(
    SetIdx      setIdx,
    ByDirection byDirection)
    const -> PartitionSpan::Bounds
{
    PartitionSpan::Bounds result{};
    result.first = mDataByPartition[setIdx].getBoundary(byDirection).operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).count;
    }
    return result;
}

auto PartitionSpan::getBoundsBoundary(
    SetIdx      setIdx,
    ByDirection byDirection,
    ByDomain    byDomain)
    const -> PartitionSpan::Bounds
{
    PartitionSpan::Bounds result{};
    result.first = mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).count;
    return result;
}

auto PartitionSpan::getGhostBoundary(SetIdx setIdx, ByDirection byDirection)
    const -> PartitionSpan::Bounds
{
    PartitionSpan::Bounds result{};
    result.first = mDataByPartition[setIdx].getGhost(byDirection).operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).count;
    }
    return result;
}

auto PartitionSpan::getGhostBoundary(
    SetIdx      setIdx,
    ByDirection byDirection,
    ByDomain    byDomain)
    const -> PartitionSpan::Bounds
{
    PartitionSpan::Bounds result{};
    result.first = mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).count;
    return result;
}

auto PartitionSpan::getGhostTarget(SetIdx setIdx, ByDirection byDirection)
    const -> PartitionSpan::GhostTarget
{
    return mDataByPartition[setIdx].getGhost(byDirection).getGhost();
}

auto PartitionSpan::getLocalPointOffset(
    SetIdx          setIdx,
    const int32_3d& point) const -> int32_t
{
    auto findings = getPossiblyLocalPointOffset(setIdx, point);
    if (std::get<0>(findings)) {
        auto classificationOffset = getClassificationOffset(setIdx,
                                                            std::get<2>(findings),
                                                            std::get<3>(findings),
                                                            std::get<4>(findings));
        return std::get<1>(findings) + classificationOffset;
    }
    NEON_THROW_UNSUPPORTED_OPERATION("Inconsistent data or query");
}


auto PartitionSpan::getPossiblyLocalPointOffset(
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

auto PartitionSpan::getClassificationOffset(
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

auto PartitionSpan::getNeighbourOfInternalPoint(
    SetIdx          setIdx,
    const int32_3d& point,
    const int32_3d& nghOffset)
    const -> int32_t
{
    // Neighbours of internal points can be internal or boundary
    auto nghPoint = point + nghOffset;
    return getLocalPointOffset(setIdx, nghPoint);
}

auto PartitionSpan::getNeighbourOfBoundaryPoint(
    SetIdx          setIdx,
    const int32_3d& point,
    const int32_3d& nghOffset)
    const -> int32_t
{
    // Neighbours of internal points can be internal or boundary
    auto nghPoint = point + nghOffset;
    auto findings = getPossiblyLocalPointOffset(setIdx, nghPoint);
    if (std::get<0>(findings)) {
        auto classificationOffset = getClassificationOffset(setIdx,
                                                            std::get<2>(findings),
                                                            std::get<3>(findings),
                                                            std::get<4>(findings));
        return std::get<1>(findings) + classificationOffset;
    }
    // We need to search on local partitions
    // We select the target partition based on the .z component of the offset
    int          partitionOffset = nghOffset.z > 0 ? +1 : -1;
    Neon::SetIdx nghSetIdx = (setIdx.idx() + mCountXpu + partitionOffset) % mCountXpu;

    findings = getPossiblyLocalPointOffset(nghSetIdx, nghPoint);
    if (std::get<0>(findings)) {
        // Ghost direction is the opposite w.r.t. the neighbour partition direction
        ByDirection ghostByDirection = nghOffset.z > 0
                                           ? ByDirection::down
                                           : ByDirection::up;
        ByDomain    ghostByDomain = std::get<4>(findings);

        Bounds ghostBounds = getGhostBoundary(setIdx,
                                              ghostByDirection,
                                              ghostByDomain);

        return std::get<1>(findings) + ghostBounds.first;
    }
    NEON_THROW_UNSUPPORTED_OPERATION("Inconsistent data or query");
}

auto PartitionSpan::allocateBlockOriginMemSet(
    Neon::Backend const& backend,
    int                  stream)
    const -> Neon::set::MemSet_t<Neon::int32_3d>
{
    Neon::MemoryOptions memOptionsAoS(
        Neon::DeviceType::CPU,
        Neon::Allocator::MALLOC,
        Neon::DeviceType::CUDA,
        backend.devType() == Neon::DeviceType::CUDA
            ? Neon::Allocator::CUDA_MEM_DEVICE
            : Neon::Allocator::NULL_MEM,
        Neon::MemoryLayout::arrayOfStructs);

    // Multi-XPU vector of Block origins (O.x,O.y,O.z)
    auto originsMemSet = backend.devSet().template newMemSet<Neon::int32_3d>(
        Neon::DataUse::IO_COMPUTE,
        1,
        memOptionsAoS,
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


}  // namespace Neon::domain::internal::experimental::bGrid::details
