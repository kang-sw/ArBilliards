#include "PxPhysicsAPI.h"

using namespace physx;
namespace {
PxDefaultErrorCallback gDefaultErrorCallback;
PxDefaultAllocator     gDefaultAllocatorCallback;
PxFoundation*          gFoundation;
} // namespace

int main(void)
{
    auto mFoundation = PxCreateFoundation(
      PX_PHYSICS_VERSION, gDefaultAllocatorCallback, gDefaultErrorCallback);
    gFoundation = mFoundation;

    if (!mFoundation)
        throw;

    bool recordMemoryAllocations = true;

    auto            mPvd      = PxCreatePvd(*gFoundation);
    PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate("127.0.0.1", 5425, 10);
    mPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);

    auto mPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *mFoundation,
                                    PxTolerancesScale(), recordMemoryAllocations, mPvd);
    if (!mPhysics)
        throw;

    auto actor = mPhysics->createRigidDynamic(PxTransform());
    PxTransform relPose(PxQuat(PxHalfPi, PxVec3(0, 0, 1)));
    auto        captureShape = PxRigidActorExt::createExclusiveShape(*actor, PxSphereGeometry(23.0), );

    mPhysics->release();
    mFoundation->release();
}