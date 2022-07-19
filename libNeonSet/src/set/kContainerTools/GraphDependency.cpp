#pragma once

#include "Neon/set/ContainerTools/GraphDependency.h"

namespace Neon::set::container {

GraphDependency::GraphDependency()
{
    mType = GraphDependencyType::NOTSET;
}

auto GraphDependency::setType(GraphDependencyType type) -> void
{
    mType = type;
}

auto GraphDependency::getType() -> GraphDependencyType
{
    return mType;
}
GraphDependency::GraphDependency(GraphDependencyType type)
{
    setType(type);
}

}  // namespace Neon::set::container
