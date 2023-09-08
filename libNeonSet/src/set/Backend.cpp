#include "Neon/set/Backend.h"
#include <cassert>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include <cstring>

#include "Neon/set/DevSet.h"

#include "mpi.h"
#include <stdexcept>

namespace Neon {

auto Backend::selfData() -> Data_t&
{
    return *m_data;
}

auto Backend::selfData() const -> const Data_t&
{
    return *m_data;
}

Backend::Backend()
{
    m_data = std::make_shared<Data_t>();
	selfData().distributed = false;
    selfData().runtime = Neon::Runtime::none;
    selfData().devSet = std::make_shared<Neon::set::DevSet>();
    selfData().streamSetVec = std::vector<Neon::set::StreamSet>(0);
    selfData().eventSetVec = std::vector<Neon::set::GpuEventSet>(0);
}

Backend::Backend(int nGpus, Neon::Runtime runtime)
{
    std::vector<int> devIds;
    for (int i = 0; i < nGpus; i++) {
        devIds.push_back(i);
    }
    m_data = std::make_shared<Data_t>();
	selfData().distributed = false;
    selfData().runtime = runtime;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(const std::vector<int>& devIds,
                 Neon::Runtime           runtime)
{
    m_data = std::make_shared<Data_t>();
	selfData().distributed = false;
    selfData().runtime = runtime;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(const Neon::set::DevSet& devSet,
                 Neon::Runtime            runtime)
{

    m_data = std::make_shared<Data_t>();
	selfData().distributed = false;
    selfData().runtime = runtime;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devSet);
    selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(const Neon::set::DevSet&    devSet,
                 const Neon::set::StreamSet& streamSet)
{
    m_data = std::make_shared<Data_t>();
	selfData().distributed = false;
    selfData().runtime = Neon::Runtime::stream;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devSet);
    selfData().streamSetVec.push_back(streamSet);
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(const std::vector<int>&     devIds,
                 const Neon::set::StreamSet& streamSet)
{
    m_data = std::make_shared<Data_t>();
	selfData().distributed = false;
    selfData().runtime = Neon::Runtime::stream;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    selfData().streamSetVec.push_back(streamSet);
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

Backend::Backend(int nGpus, int argc, char* argv[]) // For distributed systems.	
{
    std::vector<int> devIds;
    for (int i = 0; i < nGpus; i++) {
        devIds.push_back(i);
    }
    m_data = std::make_shared<Data_t>();
	selfData().distributed = true;
	selfData().numDev = nGpus;
	// selfData().sizeDeviceMem = sizeMem;
    selfData().runtime = Neon::Runtime::stream;
    selfData().devSet = std::make_shared<Neon::set::DevSet>(devType(), devIds);
    selfData().streamSetVec.push_back(selfData().devSet->defaultStreamSet());
    h_initFirstEvent();
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());

	// Initializing MPI:
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &(selfData().myRank));
	MPI_Comm_size(MPI_COMM_WORLD, &(selfData().numRank));

	selfData().communicators.resize(selfData().numRank);

	// We assume ranks to start from 0 and contiguous:
	if (selfData().myRank >= selfData().numRank) { // TODO: Better error report (e.g. NEON_EXCEPTION...).
		//std::cerr << "MPI Ranks Not As Expected" << std::endl;
		//return 1; // Indicate an error occurred by returning a non-zero value.
		throw std::runtime_error("MPI Ranks Not As Expected"); // TODO: Whenever construct, need to catch! TODO: Maybe just exit(1) with the risk of memory leak?
	}
		
	gethostname(selfData().hostname, 1024);
	for (int i = 0; i < 1024; i++) {
		if (selfData().hostname[i] == '.') { // TODO: Ask why stop after a dot?
			selfData().hostname[i] = '\0';
			return;
		}
	}

	// Set local rank (local rank is for assigning distinct GPUs to different processes):
	char hostnames[selfData().numRank * 1024]; // Better do 1D than 2D because a parameter in MPI_Allgather.
	strncpy(hostnames + selfData().myRank * 1024, selfData().hostname, 1024); // Assume MPI ranks contiguous and start from 0.
	hostnames[selfData().myRank * 1024 + 1024 - 1] = '\0';
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostnames, 1024, MPI_CHAR, MPI_COMM_WORLD);

	selfData().localRank = 0;
	int resultStrncmp;
	for (int p = 0; p < selfData().numRank; p++) {
		if (p == selfData().myRank) break;
			resultStrncmp = strncmp(hostnames + p * 1024, hostnames + selfData().myRank * 1024, std::min(strnlen(hostnames + p * 1024, 1024), strnlen(hostnames + selfData().myRank * 1024, 1024))); // Careful can't just be 1024, as we only care about chars before first null-terminator.
		
		if (resultStrncmp == 0) {
			selfData().localRank++;
		}
	}

	// Set up device memory:
	selfData().sendBuff = (float**)malloc(nGpus * sizeof(float*));
	selfData().recvBuff = (float**)malloc(nGpus * sizeof(float*));

//	for (int i = 0; i < nGpus; ++i) { // ++i and i++ no logical difference here, but ++i faster when number of iterations large according to "https://bytes.com/topic/c/answers/763572-difference-between-i-i-loop".
//		CUDACHECK(cudaSetDevice(selfData().localRank * nGpus + i));
//		CUDACHECK(cudaMalloc(selfData().sendbuff + i, sizeMem * sizeof(float)));
//		CUDACHECK(cudaMalloc(selfData().recvbuff + i, sizeMem * sizeof(float)));
//		CUDACHECK(cudaMemset(selfData().sendbuff[i], 1, sizeMem * sizeof(float)));
//		CUDACHECK(cudaMemset(selfData()recvbuff[i], 0, sizeMem * sizeof(float))); // TODO: Why one sets to "1", the other sets to "0"?
//	} // Max: Memory are allocated when creating field so no need here.

	// Set up NCCL:
	// Generating NCCL unique ID at one process and broadcasting it to all:
	if (selfData().myRank == 0) {
		ncclGetUniqueId(&(selfData().ncclId));
	}
	MPI_Bcast((void *) &(selfData().ncclId), sizeof(selfData().ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);

	// Initializing NCCL, group API is required around ncclCommInitRank as it is called across multiple GPUs in each thread/process:
	ncclGroupStart();
	for (int i = 0; i < selfData().numRank - 1; ++i) { // (numRank - 1) many communicators.
		if (selfData().myRank == i || selfData().myRank == i + 1) {
			int offsetRank = selfData().myRank == i ? selfData().numDev - 1 : 0;
			cudaSetDevice(offsetRank);
			ncclCommInitRank(&(selfData().communicators[i]), selfData().numRank * selfData().numDev, selfData().ncclId, selfData().myRank * selfData().numDev + offsetRank); // Create communicators for neighbor exchange (i.e. one communicator for adjacent pair of processes).		
		}
	}
	ncclGroupEnd();
}
	

auto Backend::clone(Neon::Runtime runtime) -> Backend
{
    Backend cloned;
    cloned.m_data = m_data;
    if (runtime != Neon::Runtime::system) {
        cloned.m_data->runtime = runtime;
    }
    return cloned;
}

auto Backend::h_initFirstEvent() -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            selfData().eventSetVec = std::vector<Neon::set::GpuEventSet>(1);
            return;
        }
        case Neon::Runtime::stream: {
            const bool disableTimingOption = true;
            selfData().eventSetVec = std::vector<Neon::set::GpuEventSet>(1, selfData().devSet->newEventSet(disableTimingOption));
            return;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto Backend::devType()
    const
    -> Neon::DeviceType
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return Neon::DeviceType::CPU;
        }
        case Neon::Runtime::stream: {
            return Neon::DeviceType::CUDA;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPERATION();
        }
    }
}

auto Backend::devSet()
    const
    -> const Neon::set::DevSet&
{
    return *selfData().devSet.get();
}

auto Backend::runtime()
    const
    -> const Neon::Runtime&
{
    return selfData().runtime;
}

auto Backend::streamSet(Neon::StreamIdx streamIdx) const -> const Neon::set::StreamSet&
{
    return selfData().streamSetVec.at(streamIdx);
}

auto Backend::streamSet(Neon::StreamIdx streamIdx) -> Neon::set::StreamSet&
{
    return selfData().streamSetVec.at(streamIdx);
}

auto Backend::eventSet(Neon::EventIdx eventdx) const -> const Neon::set::GpuEventSet&
{
    return selfData().userEventSetVec.at(eventdx);
}

auto Backend::eventSet(Neon::EventIdx eventdx) -> Neon::set::GpuEventSet&
{
    return selfData().userEventSetVec.at(eventdx);
}

// auto Backend::streamSetRotate(int&                    rotateIdx,
//                               const std::vector<int>& streamIdxVec)
//     const
//     -> const Neon::set::StreamSet&
//{
//     int streamIdx = Backend::streamSetIdxRotate(rotateIdx, streamIdxVec);
//     return selfData().streamSetVec.at(streamIdx);
// }
//
// auto Backend::streamSetIdxRotate(int&                    rotateIdx,
//                                  const std::vector<int>& streamIdxVec)
//     -> int
//{
//     int streamIdx = streamIdxVec.at(rotateIdx);
//     rotateIdx++;
//     rotateIdx = rotateIdx % streamIdxVec.size();
//     return streamIdx;
// }

auto Backend::pushEventOnStream(int eventId, int streamId)
    -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return;
        }
        case Neon::Runtime::stream: {
            break;
            ;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }

    try {
        auto& streamSet = selfData().streamSetVec.at(streamId);
        try {
            auto event = selfData().userEventSetVec.at(eventId);
            streamSet.enqueueEvent(event);
        } catch (...) {
            NeonException exp("pushEventOnStream");
            exp << "Error trying to enqueue an event";
            NEON_THROW(exp);
        }
    } catch (...) {
        NeonException exp("pushEventOnStream");
        exp << "Error trying to extract a stream";
        NEON_THROW(exp);
    }
}

auto Backend::pushEventOnStream(Neon::SetIdx setIdx,
                                int          eventId,
                                int          streamId)
    -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return;
        }
        case Neon::Runtime::stream: {
            break;
            ;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }

    try {
        auto& streamSet = selfData().streamSetVec.at(streamId);
        try {
            auto event = selfData().userEventSetVec.at(eventId);
            streamSet.enqueueEvent(setIdx, event);
        } catch (...) {
            NeonException exp("pushEventOnStream");
            exp << "Error trying to enqueue an event";
            NEON_THROW(exp);
        }
    } catch (...) {
        NeonException exp("pushEventOnStream");
        exp << "Error trying to extract a stream";
        NEON_THROW(exp);
    }
}

auto Backend::waitEventOnStream(int eventId, int streamId)
    -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return;
        }
        case Neon::Runtime::stream: {
            break;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    try {
        selfData().streamSetVec.at(streamId).waitForEvent(selfData().userEventSetVec.at(eventId));
    } catch (...) {
        NeonException exp("waitEventOnStream");
        exp << "Error trying to wait for stream";
        NEON_THROW(exp);
    }
}

auto Backend::waitEventOnStream(Neon::SetIdx setIdx, int eventId, int streamId)
    -> void
{
    switch (selfData().runtime) {
        case Neon::Runtime::openmp: {
            return;
        }
        case Neon::Runtime::stream: {
            break;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    try {
        selfData().streamSetVec.at(streamId).waitForEvent(setIdx, selfData().userEventSetVec.at(eventId));
    } catch (...) {
        NeonException exp("waitEventOnStream");
        exp << "Error trying to wait for stream";
        NEON_THROW(exp);
    }
}

// auto Backend::streamEventBarrier(const std::vector<int>& streamIdxVec) -> void
//{
//     switch (selfData().runtime) {
//         case Neon::Runtime::openmp: {
//             return;
//         }
//         case Neon::Runtime::stream: {
//             break;
//             ;
//         }
//         default: {
//             NEON_THROW_UNSUPPORTED_OPTION("");
//         }
//     }
//
//     for (int i = 0; i < int(streamIdxVec.size()); i++) {
//         selfData().streamSetVec.at(i).enqueueEvent(selfData().eventSetVec.at(i));
//     }
//
//
//     auto nextPowerOfTwo = [](unsigned int n) -> unsigned int {
//         // if n=4 v=4
//         // if n=5 v=8
//         int v = n;
//
//         v--;
//         v |= v >> 1;
//         v |= v >> 2;
//         v |= v >> 4;
//         v |= v >> 8;
//         v |= v >> 16;
//         v++;  // next power of 2
//
//         return v;
//     };
//     const unsigned int nstreams = (unsigned int)(streamIdxVec.size());
//     const unsigned int nstreamsPow2 = nextPowerOfTwo(nstreams);
//
//     for (int rLevel = 1; rLevel < int(nstreamsPow2); rLevel *= 2) {
//         const int waiterJump = rLevel * 2;
//         const int neighbourToWaitJump = rLevel;
//         for (int waiterId = 0; waiterId < int(nstreams); waiterId += waiterJump) {
//             const int neighbourId = waiterId + neighbourToWaitJump;
//             if (neighbourId < int(nstreams)) {
//                 int waiterStreamId = streamIdxVec[waiterId];
//                 int neighbourEventId = streamIdxVec[neighbourId];
//
//                 auto waiterStream = selfData().streamSetVec.at(waiterStreamId);
//                 auto neighborEvent = selfData().eventSetVec.at(neighbourEventId);
//                 waiterStream.waitForEvent(neighborEvent);
//                 // std::cout << " waiterStreamId " << waiterStreamId <<" neighbourEventId " <<neighbourEventId<<std::endl;
//             }
//         }
//     }
//
//     return;
// }

auto Backend::setAvailableStreamSet(int nStreamSets) -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        selfData().streamSetVec = std::vector<Neon::set::StreamSet>(nStreamSets);
        selfData().eventSetVec = std::vector<Neon::set::GpuEventSet>(nStreamSets);
        return;
    }
    const int streamsToAdd = nStreamSets - int(selfData().streamSetVec.size());
    for (int i = 0; i < streamsToAdd; i++) {
        selfData().streamSetVec.push_back(selfData().devSet->newStreamSet());
        const bool disableTimingOption = true;
        selfData().eventSetVec.push_back(selfData().devSet->newEventSet(disableTimingOption));
    }
    assert(selfData().eventSetVec.size() == selfData().streamSetVec.size());
}

auto Backend::setAvailableUserEvents(int nUserEventSets) -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        selfData().userEventSetVec = std::vector<Neon::set::GpuEventSet>(nUserEventSets);
        return;
    }
    const int eventToAdd = nUserEventSets - int(selfData().userEventSetVec.size());
    for (int i = 0; i < eventToAdd; i++) {
        const bool disableTimingOption = true;
        selfData().userEventSetVec.push_back(selfData().devSet->newEventSet(disableTimingOption));
    }
}


auto Backend::sync() const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        return selfData().streamSetVec[0].sync();
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync() not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::syncAll() const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        int nStreamSetVec = int(selfData().streamSetVec.size());
        for (int i = 0; i < nStreamSetVec; i++) {
            selfData().streamSetVec[i].sync();
        }
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::syncAll() not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

// auto Backend::syncAllDistributed() const -> void // TODO: Likely just merge with <syncAll> to hide distributed systems logic at backend.
// {
// 	if (runtime() == Neon::Runtime::openmp) {
// 		return; // TODO: Not handled because not sure why openmp involved.	
// 	}
// 	if (runtime() == Neon::Runtime::stream) {
// 		int nStreamSetVec = int(selfData().streamSetVec.size());
// 		for (int i = 0; i < nStreamSetVec; i++) { // Local devices/GPUs sync.
// 			selfData().streamSetVec[i].sync(); // Sync a <StreamSet> object.
// 		}
// 		MPI_Barrier(MPI_COMM_WORLD); // Global processes sync.	
// 	}
//     NeonException exp("BackendConfig_t");
//     exp << "Backend::syncAll() not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
//     NEON_THROW(exp);
// 
// }


auto Backend::sync(int idx) const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        selfData().streamSetVec[idx].sync();
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync with idx not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::sync(Neon::SetIdx setIdx, int idx) const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        selfData().streamSetVec[idx].sync(setIdx.idx());
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync with idx not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::syncEvent(Neon::SetIdx setIdx, int eventIdx) const -> void
{
    if (runtime() == Neon::Runtime::openmp) {
        return;
    }
    if (runtime() == Neon::Runtime::stream) {
        const Neon::set::GpuEventSet& eventSet = selfData().userEventSetVec.at(eventIdx);
        eventSet.sync(setIdx);
        return;
    }
    NeonException exp("BackendConfig_t");
    exp << "Backend::sync with idx not permitted for a " << Neon::RuntimeUtils::toString(runtime()) << "backend";
    NEON_THROW(exp);
}

auto Backend::runMode() const -> const Neon::run_et ::et&
{
    return selfData().runMode;
}

auto Backend::runMode(Neon::run_et::et runMode) -> void
{
    selfData().runMode = runMode;
}


std::string Backend::toString(Neon::Runtime e)
{
    switch (e) {
        case Neon::Runtime::none: {
            return "none";
        }
        case Neon::Runtime::stream: {
            return "stream";
        }
        case Neon::Runtime::openmp: {
            return "openmp";
        }
        default: {
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

std::string Backend::toString() const
{
    std::ostringstream msg;
    msg << "Backend (" << this << ") - [runtime:" << toString(selfData().runtime) << "] [nDev:" << selfData().devSet->setCardinality() << "] ";
    switch (selfData().devSet->type()) {
        case Neon::DeviceType::OMP:
        case Neon::DeviceType::CPU: {
            for (int i = 0; i < selfData().devSet->setCardinality(); i++) {
                msg << "[dev" << i << ":" << selfData().devSet->devId(i) << "] ";
            }
            break;
        }
        case Neon::DeviceType::CUDA: {
            for (int i = 0; i < selfData().devSet->setCardinality(); i++) {
                msg << "[dev" << i << ":" << selfData().devSet->devId(i) << " " << selfData().devSet->getDevName(i) << "] ";
            }
            break;
        }
            //        case dev_et::NONE:
            //        case dev_et::MPI:
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    return msg.str();
}

auto Backend::getMemoryOptions(Neon::MemoryLayout order) const
    -> Neon::MemoryOptions
{
    MemoryOptions memoryOptions(Neon::DeviceType::CPU,
                                devSet().type(),
                                order);
    return memoryOptions;
}

auto Backend::getMemoryOptions(Neon::Allocator    ioAllocator,
                               Neon::Allocator    computeAllocators[Neon::DeviceTypeUtil::nConfig],
                               Neon::MemoryLayout order) const
    -> Neon::MemoryOptions
{
    MemoryOptions memoryOptions(Neon::DeviceType::CPU,
                                ioAllocator,
                                devSet().type(),
                                computeAllocators,
                                order);
    return memoryOptions;
}

auto Backend::getMemoryOptions() const
    -> Neon::MemoryOptions
{
    const Neon::MemoryOptions defaultMemOption = getMemoryOptions(Neon::MemoryLayout::structOfArrays);
    return defaultMemOption;
}

auto Backend::toReport(Neon::Report& report, Report::SubBlock* subdocAPI) const -> void
{
    Report::SubBlock* targetSubDoc = subdocAPI;
    Report::SubBlock  tmp;
    if (nullptr == subdocAPI) {
        tmp = report.getSubdoc();
        targetSubDoc = &tmp;
    }

    report.addMember("Runtime", Neon::RuntimeUtils::toString(runtime()), targetSubDoc);
    report.addMember("DeviceType", Neon::DeviceTypeUtil::toString(devType()), targetSubDoc);
    report.addMember("NumberOfDevices", devSet().setCardinality(), targetSubDoc);
    report.addMember(
        "Devices", [&] {
            std::vector<int> idsList;
            for (auto const& id : devSet().devId()) {
                idsList.push_back(id.idx());
            }
            return idsList;
        }(),
        targetSubDoc);

    if (nullptr == subdocAPI) {
        report.addSubdoc("Backend", *targetSubDoc);
    }
}

auto Backend::getDeviceCount() const -> int
{
    return m_data->devSet->setCardinality();
}

auto Backend::helpDeviceToDeviceTransferByte(int                     streamId,
                                             size_t                  bytes,
                                             Neon::set::TransferMode transferMode,
                                             Neon::SetIdx            dstSet,
                                             char*                   dstAddr,
                                             Neon::SetIdx            srcSet,
                                             const char*             srcAddr) const -> void
{
    if (bytes == 0) {
        return;
    }
    auto& devSet = this->devSet();
    auto& targetStreamSet = streamSet(streamId);

    devSet.transfer(transferMode,
                    targetStreamSet,
                    dstSet,
                    dstAddr,
                    srcSet,
                    srcAddr,
                    bytes);
}

auto Backend::helpNodeToNodeTransferByte(int 			streamIdx, 
										size_t 			sizeTransfer,
										Neon::SetIdx	srcIdx,
										int 			targetRank, 
										const char* 	sendBuff, 
										char*			recvBuff,
										ncclComm_t		communicator) const -> void
{
	// A transfer consists of two endpoints. In our logic, the exchange is symmetric, i.e. A sends to B iff B sends to A. Therefore, when A sends to B via ncclSend(), we can immediately do A receives from B via ncclRecv().
	
	if (sizeTransfer == 0) { // In case transfer size is 0.
		return;
	}

    auto& stream = streamSet(streamIdx)[srcIdx].stream(); // TODO: Not sure how stream mechanism in Neon work, above is mimicking deviceToDeviceTransfer if you dig deep enough.

	ncclGroupStart();
	// No need to cudaSetDevice() because we carefully initialize communicators in backend constructor overcome this.
	ncclSend(sendBuff, sizeTransfer, ncclChar, targetRank, communicator, stream); // TODO: Took shortcut here and assume it is ncclChar we work with because in run() in DataTransferContainer this function is called with char type.
	ncclRecv(recvBuff, sizeTransfer, ncclChar, targetRank, communicator, stream);
	ncclGroupEnd();	
}

auto Backend::deviceCount() const -> int
{
    return devSet().setCardinality();
}

auto Backend::isFirstDevice(Neon::SetIdx id) const -> bool
{
    return id.idx() == 0;
}
auto Backend::isLastDevice(Neon::SetIdx id) const -> bool
{
    return id.idx() == (deviceCount() - 1);
}
}  // namespace Neon
