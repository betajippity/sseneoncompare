#include "sseneoncompare.hpp"

/* A direct implementation of "An Efficient and Robust Ray-Box Intersection Algorithm" by
   Amy Williams et al. 2005; DOI: 10.1080/2151237X.2005.10129188 */
bool rayBBoxIntersectScalar(const Ray& ray, const BBox& bbox, float& tMin, float& tMax) {
    FVec4 rdir = 1.0f / ray.direction;
    int sign[3];
    sign[0] = (rdir.x < 0);
    sign[1] = (rdir.y < 0);
    sign[2] = (rdir.z < 0);

    float tyMin, tyMax, tzMin, tzMax;
    tMin = (bbox.cornersAlt[sign[0]][0] - ray.origin.x) * rdir.x;
    tMax = (bbox.cornersAlt[1 - sign[0]][0] - ray.origin.x) * rdir.x;
    tyMin = (bbox.cornersAlt[sign[1]][1] - ray.origin.y) * rdir.y;
    tyMax = (bbox.cornersAlt[1 - sign[1]][1] - ray.origin.y) * rdir.y;
    if ((tMin > tyMax) || (tyMin > tMax)) {
        return false;
    }
    if (tyMin > tMin) {
        tMin = tyMin;
    }
    if (tyMax < tMax) {
        tMax = tyMax;
    }
    tzMin = (bbox.cornersAlt[sign[2]][2] - ray.origin.z) * rdir.z;
    tzMax = (bbox.cornersAlt[1 - sign[2]][2] - ray.origin.z) * rdir.z;
    if ((tMin > tzMax) || (tzMin > tMax)) {
        return false;
    }
    if (tzMin > tMin) {
        tMin = tzMin;
    }
    if (tzMax < tMax) {
        tMax = tzMax;
    }
    return ((tMin < ray.tMax) && (tMax > ray.tMin));
}

/* A much more compact implementation of Williams et al. 2005; this implementation does not
   calculate a negative tMin if the ray origin is inside of the box. */
bool rayBBoxIntersectScalarCompact(const Ray& ray, const BBox& bbox, float& tMin, float& tMax) {
    FVec4 rdir = 1.0f / ray.direction;
    IVec4 near(int(rdir.x >= 0.0f ? 0 : 3), int(rdir.y >= 0.0f ? 1 : 4),
               int(rdir.z >= 0.0f ? 2 : 5));
    IVec4 far(int(rdir.x >= 0.0f ? 3 : 0), int(rdir.y >= 0.0f ? 4 : 1),
              int(rdir.z >= 0.0f ? 5 : 2));

    tMin = fmax(fmax(ray.tMin, (bbox.corners[near.x] - ray.origin.x) * rdir.x),
                fmax((bbox.corners[near.y] - ray.origin.y) * rdir.y,
                     (bbox.corners[near.z] - ray.origin.z) * rdir.z));
    tMax = fmin(fmin(ray.tMax, (bbox.corners[far.x] - ray.origin.x) * rdir.x),
                fmin((bbox.corners[far.y] - ray.origin.y) * rdir.y,
                     (bbox.corners[far.z] - ray.origin.z) * rdir.z));

    if (std::isnan(tMin) || std::isnan(tMax) || std::isinf(tMin) || std::isinf(tMax)) {
        return false;
    } else {
        return true;
    }
}

void rayBBoxIntersect4Scalar(const Ray& ray,
                             const BBox& bbox0,
                             const BBox& bbox1,
                             const BBox& bbox2,
                             const BBox& bbox3,
                             IVec4& hits,
                             FVec4& tMins,
                             FVec4& tMaxs) {
    hits[0] = (int)rayBBoxIntersectScalar(ray, bbox0, tMins[0], tMaxs[0]);
    hits[1] = (int)rayBBoxIntersectScalar(ray, bbox1, tMins[1], tMaxs[1]);
    hits[2] = (int)rayBBoxIntersectScalar(ray, bbox2, tMins[2], tMaxs[2]);
    hits[3] = (int)rayBBoxIntersectScalar(ray, bbox3, tMins[3], tMaxs[3]);
}

void rayBBoxIntersect4ScalarCompact(const Ray& ray,
                                    const BBox& bbox0,
                                    const BBox& bbox1,
                                    const BBox& bbox2,
                                    const BBox& bbox3,
                                    IVec4& hits,
                                    FVec4& tMins,
                                    FVec4& tMaxs) {
    hits[0] = (int)rayBBoxIntersectScalarCompact(ray, bbox0, tMins[0], tMaxs[0]);
    hits[1] = (int)rayBBoxIntersectScalarCompact(ray, bbox1, tMins[1], tMaxs[1]);
    hits[2] = (int)rayBBoxIntersectScalarCompact(ray, bbox2, tMins[2], tMaxs[2]);
    hits[3] = (int)rayBBoxIntersectScalarCompact(ray, bbox3, tMins[3], tMaxs[3]);
}

// SSE version of the compact Williams et al. 2005 implementation
void rayBBoxIntersect4SSE(const Ray& ray,
                          const BBox4& bbox4,
                          IVec4& hits,
                          FVec4& tMins,
                          FVec4& tMaxs) {
    FVec4 rdir(_mm_set1_ps(1.0f) / ray.direction.m128);
    /* use _mm_shuffle_ps, which translates to a single instruction while _mm_set1_ps involves a
       MOVSS + a shuffle */
    FVec4 rdirX(_mm_shuffle_ps(rdir.m128, rdir.m128, _MM_SHUFFLE(0, 0, 0, 0)));
    FVec4 rdirY(_mm_shuffle_ps(rdir.m128, rdir.m128, _MM_SHUFFLE(1, 1, 1, 1)));
    FVec4 rdirZ(_mm_shuffle_ps(rdir.m128, rdir.m128, _MM_SHUFFLE(2, 2, 2, 2)));
    FVec4 originX(_mm_set1_ps(ray.origin.x));
    FVec4 originY(_mm_set1_ps(ray.origin.y));
    FVec4 originZ(_mm_set1_ps(ray.origin.z));

    IVec4 near(int(rdir.x >= 0.0f ? 0 : 3), int(rdir.y >= 0.0f ? 1 : 4),
               int(rdir.z >= 0.0f ? 2 : 5));
    IVec4 far(int(rdir.x >= 0.0f ? 3 : 0), int(rdir.y >= 0.0f ? 4 : 1),
              int(rdir.z >= 0.0f ? 5 : 2));

    tMins = FVec4(_mm_max_ps(
        _mm_max_ps(_mm_set1_ps(ray.tMin), (bbox4.cornersSSE[near.x] - originX.m128) * rdirX.m128),
        _mm_max_ps((bbox4.cornersSSE[near.y] - originY.m128) * rdirY.m128,
                   (bbox4.cornersSSE[near.z] - originZ.m128) * rdirZ.m128)));
    tMaxs = FVec4(_mm_min_ps(
        _mm_min_ps(_mm_set1_ps(ray.tMax), (bbox4.cornersSSE[far.x] - originX.m128) * rdirX.m128),
        _mm_min_ps((bbox4.cornersSSE[far.y] - originY.m128) * rdirY.m128,
                   (bbox4.cornersSSE[far.z] - originZ.m128) * rdirZ.m128)));

    int hit = ((1 << 4) - 1) & _mm_movemask_ps(_mm_cmple_ps(tMins.m128, tMaxs.m128));
    hits[0] = bool(hit & (1 << (0)));
    hits[1] = bool(hit & (1 << (1)));
    hits[2] = bool(hit & (1 << (2)));
    hits[3] = bool(hit & (1 << (3)));
}

#if defined(__aarch64__)

inline uint32_t neonCompareAndMask(const float32x4_t& a, const float32x4_t& b) {
    uint32x4_t compResUint = vcleq_f32(a, b);
    static const int32x4_t shift = { 0, 1, 2, 3 };
    uint32x4_t tmp = vshrq_n_u32(compResUint, 31);
    return vaddvq_u32(vshlq_u32(tmp, shift));
}

// Neon version of the compact Williams et al. 2005 implementation
void rayBBoxIntersect4Neon(const Ray& ray,
                           const BBox4& bbox4,
                           IVec4& hits,
                           FVec4& tMins,
                           FVec4& tMaxs) {
    FVec4 rdir(vdupq_n_f32(1.0f) / ray.direction.f32x4);
    /* since NEON doesn't have a single-instruction equivalent to _mm_shuffle_ps, we just take
       the slow route here and load into each float32x4_t */
    FVec4 rdirX(vdupq_n_f32(rdir.x));
    FVec4 rdirY(vdupq_n_f32(rdir.y));
    FVec4 rdirZ(vdupq_n_f32(rdir.z));
    FVec4 originX(vdupq_n_f32(ray.origin.x));
    FVec4 originY(vdupq_n_f32(ray.origin.y));
    FVec4 originZ(vdupq_n_f32(ray.origin.z));

    IVec4 near(int(rdir.x >= 0.0f ? 0 : 3), int(rdir.y >= 0.0f ? 1 : 4),
               int(rdir.z >= 0.0f ? 2 : 5));
    IVec4 far(int(rdir.x >= 0.0f ? 3 : 0), int(rdir.y >= 0.0f ? 4 : 1),
              int(rdir.z >= 0.0f ? 5 : 2));

    tMins = FVec4(vmaxq_f32(
        vmaxq_f32(vdupq_n_f32(ray.tMin), (bbox4.cornersNeon[near.x] - originX.f32x4) * rdirX.f32x4),
        vmaxq_f32((bbox4.cornersNeon[near.y] - originY.f32x4) * rdirY.f32x4,
                  (bbox4.cornersNeon[near.z] - originZ.f32x4) * rdirZ.f32x4)));
    tMaxs = FVec4(vminq_f32(
        vminq_f32(vdupq_n_f32(ray.tMax), (bbox4.cornersNeon[far.x] - originX.f32x4) * rdirX.f32x4),
        vminq_f32((bbox4.cornersNeon[far.y] - originY.f32x4) * rdirY.f32x4,
                  (bbox4.cornersNeon[far.z] - originZ.f32x4) * rdirZ.f32x4)));

    uint32_t hit = neonCompareAndMask(tMins.f32x4, tMaxs.f32x4);
    hits[0] = bool(hit & (1 << (0)));
    hits[1] = bool(hit & (1 << (1)));
    hits[2] = bool(hit & (1 << (2)));
    hits[3] = bool(hit & (1 << (3)));
}

#endif

// Compact scalar version written to be easily autovectorized
void rayBBoxIntersect4AutoVectorize(const Ray& ray,
                                    const BBox4& bbox4,
                                    IVec4& hits,
                                    FVec4& tMins,
                                    FVec4& tMaxs) {
    float rdir[3] = { 1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z };
    float rdirX[4] = { rdir[0], rdir[0], rdir[0], rdir[0] };
    float rdirY[4] = { rdir[1], rdir[1], rdir[1], rdir[1] };
    float rdirZ[4] = { rdir[2], rdir[2], rdir[2], rdir[2] };
    float originX[4] = { ray.origin.x, ray.origin.x, ray.origin.x, ray.origin.x };
    float originY[4] = { ray.origin.y, ray.origin.y, ray.origin.y, ray.origin.y };
    float originZ[4] = { ray.origin.z, ray.origin.z, ray.origin.z, ray.origin.z };
    float rtMin[4] = { ray.tMin, ray.tMin, ray.tMin, ray.tMin };
    float rtMax[4] = { ray.tMax, ray.tMax, ray.tMax, ray.tMax };

    IVec4 near(int(rdir[0] >= 0.0f ? 0 : 3), int(rdir[1] >= 0.0f ? 1 : 4),
               int(rdir[2] >= 0.0f ? 2 : 5));
    IVec4 far(int(rdir[0] >= 0.0f ? 3 : 0), int(rdir[1] >= 0.0f ? 4 : 1),
              int(rdir[2] >= 0.0f ? 5 : 2));

    float product0[4];

#pragma clang loop vectorize(enable)
    for (int i = 0; i < 4; i++) {
        product0[i] = bbox4.cornersSSE[near.y][i] - originY[i];
        tMins[i] = bbox4.cornersSSE[near.z][i] - originZ[i];
        product0[i] = product0[i] * rdirY[i];
        tMins[i] = tMins[i] * rdirZ[i];
        product0[i] = fmax(product0[i], tMins[i]);
        tMins[i] = bbox4.cornersSSE[near.x][i] - originX[i];
        tMins[i] = tMins[i] * rdirX[i];
        tMins[i] = fmax(rtMin[i], tMins[i]);
        tMins[i] = fmax(product0[i], tMins[i]);

        product0[i] = bbox4.cornersSSE[far.y][i] - originY[i];
        tMaxs[i] = bbox4.cornersSSE[far.z][i] - originZ[i];
        product0[i] = product0[i] * rdirY[i];
        tMaxs[i] = tMaxs[i] * rdirZ[i];
        product0[i] = fmin(product0[i], tMaxs[i]);
        tMaxs[i] = bbox4.cornersSSE[far.x][i] - originX[i];
        tMaxs[i] = tMaxs[i] * rdirX[i];
        tMaxs[i] = fmin(rtMax[i], tMaxs[i]);
        tMaxs[i] = fmin(product0[i], tMaxs[i]);

        hits[i] = !(std::isnan(tMins[i]) || std::isnan(tMaxs[i]) || std::isinf(tMins[i]) ||
                    std::isinf(tMaxs[i]));
    }
}
