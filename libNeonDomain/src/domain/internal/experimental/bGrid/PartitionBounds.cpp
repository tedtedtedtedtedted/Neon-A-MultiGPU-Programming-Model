#include "Neon/domain/internal/experimental/bGrid/PartitionBounds.h"
#include "Neon/core/core.h"

namespace Neon::domain::internal::experimental::bGrid::details {

PartitionBounds::PartitionBounds(Neon::Backend const&  backend,
                                 SpanClassifier const& spanClassifier)
{
    countXpu = backend.devSet().setCardinality();
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

auto PartitionBounds::getBoundsInternal(SetIdx setIdx) -> PartitionBounds::Bounds
{
    PartitionBounds::Bounds result{};
    result.first = mDataByPartition[setIdx].getInternal().operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getInternal().operator()(byDomain).count;
    }
    return result;
}

auto PartitionBounds::getBoundsInternal(SetIdx setIdx, ByDomain byDomain) -> PartitionBounds::Bounds
{
    PartitionBounds::Bounds result{};
    result.first = mDataByPartition[setIdx].getInternal().operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getInternal().operator()(byDomain).count;
    return result;
}

auto PartitionBounds::getBoundsBoundary(SetIdx setIdx, ByDirection byDirection) -> PartitionBounds::Bounds
{
    PartitionBounds::Bounds result{};
    result.first = mDataByPartition[setIdx].getBoundary(byDirection).operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).count;
    }
    return result;
}

auto PartitionBounds::getBoundsBoundary(SetIdx setIdx, ByDirection byDirection, ByDomain byDomain) -> PartitionBounds::Bounds
{
    PartitionBounds::Bounds result{};
    result.first = mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).count;
    return result;
}

auto PartitionBounds::getGhostBoundary(SetIdx setIdx, ByDirection byDirection) -> PartitionBounds::Bounds
{
    PartitionBounds::Bounds result{};
    result.first = mDataByPartition[setIdx].getGhost(byDirection).operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).count;
    }
    return result;
}

auto PartitionBounds::getGhostBoundary(SetIdx setIdx, ByDirection byDirection, ByDomain byDomain) -> PartitionBounds::Bounds
{
    PartitionBounds::Bounds result{};
    result.first = mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).count;
    return result;
}

auto PartitionBounds::getGhostTarget(SetIdx setIdx, ByDirection byDirection) -> PartitionBounds::GhostTarget
{
    return mDataByPartition[setIdx].getGhost(byDirection).getGhost();
}


}  // namespace Neon::domain::internal::experimental::bGrid::details
