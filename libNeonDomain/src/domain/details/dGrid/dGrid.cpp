#include "Neon/domain/details/dGrid/dGrid.h"

namespace Neon::domain::details::dGrid {

dGrid::dGrid()
{
    mData = std::make_shared<Data>();
}

dGrid::Data::Data(const Neon::Backend& backend,
				  const Neon::int32_3d& dimension)
{
    partitionDims = backend.devSet().newDataSet<index_3d>({0, 0, 0});
    firstZIndex = backend.devSet().newDataSet<index_t>(0);
    spanTable = Neon::domain::tool::SpanTable<dSpan>(backend);
    elementsPerPartition = Neon::domain::tool::SpanTable<int>(backend);

    halo = index_3d(0, 0, 0);
    reduceEngine = Neon::sys::patterns::Engine::cuBlas;
	
	// Ted: Below added member for <dGrid> to find its global origins after partition-by-node/process in distributed systems:
	if (!backend.isDistributed()) {
		zOrigin = 0;
	} else {
		int uniformProc = dimension.z / backend.getProcessCount(); // TODO: Ted: Ask Max need <int32_t>?
		int reminderProc = dimension.z % backend.getProcessCount();
		if (backend.getRank() < reminderProc) {
			zOrigin = backend.getRank() * (uniformProc + 1);
		} else {
			zOrigin = reminderProc * (uniformProc + 1) + (backend.getRank() - reminderProc) * uniformProc;
		}
	}
}

auto dGrid::helpFieldMemoryAllocator()
    const -> const Neon::aGrid&
{
    return mData->memoryGrid;
}

auto dGrid::getSpan(Neon::Execution execution,
                    SetIdx          setIdx,
                    Neon::DataView  dataView)
    const -> const Span&
{
    return mData->spanTable.getSpan(execution, setIdx, dataView);
}


auto dGrid::helpGetPartitionDim()
    const -> const Neon::set::DataSet<index_3d>
{
    return mData->partitionDims;
}

// auto dGrid::helpIdexPerPartition(Neon::DataView dataView)
//     const -> const Neon::set::DataSet<int>
//{
//     return mData->elementsPerPartition.getSpan(dataView);
// }

auto dGrid::setReduceEngine(Neon::sys::patterns::Engine eng)
    -> void
{
    mData->reduceEngine = eng;
}

auto dGrid::getLaunchParameters(const Neon::DataView  dataView,
                                const Neon::index_3d& blockSize,
                                const size_t&         shareMem) const -> Neon::set::LaunchParameters
{
    Neon::set::LaunchParameters ret = getBackend().devSet().newLaunchParameters();

    auto dimsByDataView = getBackend().devSet().newDataSet<index_3d>([&](Neon::SetIdx const& setIdx,
                                                                         auto&               value) {
        value = getSpan(Neon::Execution::host, setIdx, dataView).mDim;
    });

    ret.set(Neon::sys::GpuLaunchInfo::domainGridMode,
            dimsByDataView,
            blockSize,
            shareMem);
    return ret;
}

auto dGrid::convertToNghIdx(const std::vector<Neon::index_3d>& stencilOffsets)
    const -> std::vector<NghIdx>
{
    std::vector<NghIdx> res;
    for (const auto& offset : stencilOffsets) {
        res.push_back(offset.template newType<int8_t>());
    }
    return res;
}

auto dGrid::convertToNghIdx(Neon::index_3d const& stencilOffsets)
    const -> NghIdx
{
    return stencilOffsets.template newType<int8_t>();
}

auto dGrid::isInsideDomain(const index_3d& idx) const -> bool
{
    bool isPositive = idx >= 0;
    bool isLover = idx < this->getDimension();
    return isLover && isPositive;
}

auto dGrid::getSetIdx(const Neon::index_3d& idx)
    const -> int32_t
{
    auto prop = getProperties(idx);
    if (!prop.isInside()) {
        return -1;
    }
    return prop.getSetIdx();
}

auto dGrid::getProperties(const index_3d& idx)
    const -> GridBaseTemplate::CellProperties
{
    GridBaseTemplate::CellProperties cellProperties;
    cellProperties.setIsInside(isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return cellProperties;
    }

    if (this->getDevSet().setCardinality() == 1) {
        cellProperties.init(0, DataView::INTERNAL);
    } else {
        int            zCounter = 0;
        int            zCounterPrevious = 0;
        Neon::SetIdx   setIdx;
        Neon::DataView dataView = DataView::BOUNDARY;
        for (int i = 0; i < this->getDevSet().setCardinality(); i++) {
            zCounter += mData->partitionDims[i].z;
            if (idx.z < zCounter) {
                setIdx = i;
            }
            if ((zCounterPrevious + mData->halo.z >= idx.z) &&
                (zCounter - mData->halo.z < idx.z)) {
                dataView = Neon::DataView::INTERNAL;
            }
            zCounterPrevious = zCounter;
        }
        cellProperties.init(setIdx, dataView);
    }
    return cellProperties;
}

auto dGrid::helpGetFirstZindex()
    const -> const Neon::set::DataSet<int32_t>&
{
    return mData->firstZIndex;
}

// Ted: <mData->zOrigin> is the true globabl z-index of the origin of the partitioned grid of its poccessing node/process in distributed systems.
auto dGrid::helpGetZOrigin()
	const -> int // TODO: Ted: Is const reference unnecessary here because <int> type is not much overhead to be copied-by-value?
{
	return mData->zOrigin;
}

}  // namespace Neon::domain::details::dGrid
