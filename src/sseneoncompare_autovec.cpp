#include "sseneoncompare.hpp"

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
        product0[i] = bbox4.corners[near.y][i] - originY[i];
        tMins[i] = bbox4.corners[near.z][i] - originZ[i];
        product0[i] = product0[i] * rdirY[i];
        tMins[i] = tMins[i] * rdirZ[i];
        product0[i] = fmax(product0[i], tMins[i]);
        tMins[i] = bbox4.corners[near.x][i] - originX[i];
        tMins[i] = tMins[i] * rdirX[i];
        tMins[i] = fmax(rtMin[i], tMins[i]);
        tMins[i] = fmax(product0[i], tMins[i]);

        product0[i] = bbox4.corners[far.y][i] - originY[i];
        tMaxs[i] = bbox4.corners[far.z][i] - originZ[i];
        product0[i] = product0[i] * rdirY[i];
        tMaxs[i] = tMaxs[i] * rdirZ[i];
        product0[i] = fmin(product0[i], tMaxs[i]);
        tMaxs[i] = bbox4.corners[far.x][i] - originX[i];
        tMaxs[i] = tMaxs[i] * rdirX[i];
        tMaxs[i] = fmin(rtMax[i], tMaxs[i]);
        tMaxs[i] = fmin(product0[i], tMaxs[i]);

        hits[i] = tMins[i] <= tMaxs[i];
    }
}