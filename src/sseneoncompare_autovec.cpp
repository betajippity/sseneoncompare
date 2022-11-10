#include "sseneoncompare.hpp"

// Compact scalar version written to be easily autovectorized
std::tuple<IVec4,FVec4,FVec4> rayBBoxIntersect4AutoVectorize(const Ray& ray,
                                    const BBox4& bbox4) {

    float rdirX = 1.0f / ray.direction.x;
    float rdirY = 1.0f / ray.direction.y;
    float rdirZ = 1.0f / ray.direction.z;

    IVec4 near(int(rdirX >= 0.0f ? 0 : 3), int(rdirY >= 0.0f ? 1 : 4),
               int(rdirZ >= 0.0f ? 2 : 5));
    IVec4 far(int(rdirX >= 0.0f ? 3 : 0), int(rdirY >= 0.0f ? 4 : 1),
              int(rdirZ >= 0.0f ? 5 : 2));

    IVec4 hits = IVec4();
    FVec4 tMins = FVec4();
    FVec4 tMaxs = FVec4();

#pragma clang loop vectorize(enable)
    for (int i = 0; i < 4; i++) {
        float product0 = bbox4.corners[near.y][i] - ray.origin.y;
        float tmin = bbox4.corners[near.z][i] - ray.origin.z;
        product0 = product0 * rdirY;
        tmin = rdirZ * tmin;
        product0 = fmax(product0, tmin);
        tmin = bbox4.corners[near.x][i] - ray.origin.x;
        tmin = tmin * rdirX;
        tmin = fmax(ray.tMin, tmin);
        tmin = fmax(product0, tmin);

        product0 = bbox4.corners[far.y][i] - ray.origin.y;
        float tmax = bbox4.corners[far.z][i] - ray.origin.z;
        product0 = product0 * rdirY;
        tmax = tmax * rdirZ;
        product0 = fmin(product0, tmax);
        tmax = bbox4.corners[far.x][i] - ray.origin.x;
        tmax = tmax * rdirX;
        tmax = fmin(ray.tMax, tmax);
        tmax = fmin(product0, tmax);

        tMaxs[i] = tmax;
        tMins[i] = tmin;
        hits[i] = tMins[i] <= tMaxs[i];
    }
    return std::make_tuple(hits,tMins,tMaxs);
}
