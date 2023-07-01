#pragma once

namespace Neon {

template <typename T>
auto Backend::newDataSet()
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs);
    return result;
}

template <typename T>
auto Backend::newDataSet(T const& val)
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs, val);
    return result;
}

template <typename T, typename Lambda>
auto Backend::newDataSet(Lambda lambda)
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs);
    result.forEachSeq(lambda);
    return result;
}

template <typename Lambda>
auto Backend::forEachDeviceSeq(const Lambda& lambda)
    const -> void
{
    int const nDevs = getDeviceCount();
    for (int i = 0; i < nDevs; i++) {
        lambda(Neon::SetIdx(i));
    }
}

template <typename Lambda>
auto Backend::forEachDevicePar(const Lambda& lambda)
    const -> void
{
    int const nDevs = getDeviceCount();
#pragma omp parallel for num_threads(nDevs) shared(lambda, nDevs) default(none)
    for (int i = 0; i < nDevs; i++) {
        lambda(Neon::SetIdx(i));
    }
}

template <typename T>
auto Backend::deviceToDeviceTransfer(int                     streamId,
                                     size_t                  nItems,
                                     Neon::set::TransferMode transferMode,
                                     Neon::SetIdx            dstSet,
                                     T*                      dstAddr,
                                     Neon::SetIdx            srcSet,
                                     T const*                srcAddr) const -> void
{
    helpDeviceToDeviceTransferByte(streamId,
                                   sizeof(T) * nItems,
                                   transferMode,
                                   dstSet,
                                   dstAddr,
                                   srcSet,
                                   srcAddr);
}

template <typename T>
auto nodeToNodeTransfer(int 			streamIdx, 
						size_t 			sizeTransfer,
						Neon::SetIdx	srcIdx
						int 			targetRank, 
						T* 				sendBuff, 
						T* 				recvBuff,
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

}  // namespace Neon
