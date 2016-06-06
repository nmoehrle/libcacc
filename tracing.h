#ifndef CACC_TRACING_HEADER
#define CACC_TRACING_HEADER

#include "bvh_tree.h"

CACC_NAMESPACE_BEGIN

#define TRACING_NAMESPACE_BEGIN namespace tracing {
#define TRACING_NAMESPACE_END }

TRACING_NAMESPACE_BEGIN

//SSTACK_SIZE * TRACE_BLOCK_SIZE has to be less than smem
#define TRACE_BLOCK_SIZE 128
#define SSTACK_SIZE 6
#define GSTACK_SIZE 64

constexpr uint NAI = (uint) -1;

void load_textures(cacc::BVHTree<cacc::DEVICE>::Data const bvh_tree);

__device__
void trace(cacc::BVHTree<cacc::DEVICE>::Data const bvh_tree,
        cacc::Ray const ray, uint * hit_face_id_ptr);

TRACING_NAMESPACE_END

CACC_NAMESPACE_END

#endif /* CACC_TRACING_HEADER */
