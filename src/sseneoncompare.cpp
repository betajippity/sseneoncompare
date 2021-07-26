#include <iostream>
#include <cmath>
#include <chrono>

#if defined(__x86_64__)
#include <xmmintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#include "sse2neon/sse2neon.h"
#endif

struct Timer {
    std::chrono::high_resolution_clock::time_point m_startTime;

    Timer() { m_startTime = std::chrono::high_resolution_clock::now(); }
    void start() { m_startTime = std::chrono::high_resolution_clock::now(); }

    int getElapsedMicoSec() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return (int)std::min(
            (int)std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_startTime)
                .count(),
            std::numeric_limits<int>::max());
    }
};

struct FVec4 {
    union {  // Use union for type punning __m128 and float32x4_t
        __m128 m128;
#if defined(__aarch64__)
        float32x4_t f32x4;
#endif
        struct {
            float x;
            float y;
            float z;
            float w;
        };
        float data[4];
    };

    FVec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
#if defined(__x86_64__)
    FVec4(__m128 f4) : m128(f4) {}
#elif defined(__aarch64__)
    FVec4(float32x4_t f4) : f32x4(f4) {}
#endif

    FVec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    FVec4(float x_, float y_, float z_) : x(x_), y(y_), z(z_), w(0.0f) {}

    float operator[](int i) const { return data[i]; }
    float& operator[](int i) { return data[i]; }

    FVec4 operator+(const FVec4& b) { return FVec4(x + b.x, y + b.y, z + b.z, w + b.w); }
    FVec4 operator-(const FVec4& b) { return FVec4(x - b.x, y - b.y, z - b.z, w - b.w); }
    FVec4 operator*(const float& b) { return FVec4(x * b, y * b, z * b, w * b); }
    FVec4 operator/(const float& b) { return FVec4(x / b, y / b, z / b, w / b); }
};

FVec4 operator/(const float& a, const FVec4& b) {
    return FVec4(a / b.x, a / b.y, a / b.z, a / b.w);
}

struct IVec4 {
    union {
        struct {
            int x;
            int y;
            int z;
            int w;
        };
        int data[4];
    };

    IVec4() : x(0), y(0), z(0), w(0) {}
    IVec4(int x_, int y_, int z_, int w_) : x(x_), y(y_), z(z_), w(w_) {}
    IVec4(int x_, int y_, int z_) : x(x_), y(y_), z(z_), w(0) {}

    int operator[](int i) const { return data[i]; }
    int& operator[](int i) { return data[i]; }
};

struct BBox {
    union {
        float corners[6];        // indexed as [minX minY minZ maxX maxY maxZ]
        float cornersAlt[2][3];  // indexed as corner[minOrMax][XYZ]
    };

    BBox(const FVec4& minCorner, const FVec4& maxCorner) {
        cornersAlt[0][0] = std::min(minCorner.x, maxCorner.x);
        cornersAlt[0][1] = std::min(minCorner.y, maxCorner.y);
        cornersAlt[0][2] = std::min(minCorner.z, maxCorner.z);
        cornersAlt[1][0] = std::max(minCorner.x, maxCorner.x);
        cornersAlt[1][1] = std::max(minCorner.y, maxCorner.y);
        cornersAlt[1][2] = std::max(minCorner.x, maxCorner.x);
    }

    FVec4 minCorner() const { return FVec4(corners[0], corners[1], corners[2]); }

    FVec4 maxCorner() const { return FVec4(corners[3], corners[4], corners[5]); }
};

struct BBox4 {
    union {
        __m128 cornersSSE[6];  // order: minX, minY, minZ, maxX, maxY, maxZ
#if defined(__aarch64__)
        float32x4_t cornersNeon[6];
#endif
        float cornersFloat[2][3][4];  // indexed as corner[minOrMax][XYZ][bboxNumber]
    };

    inline __m128* minCornerSSE() { return &cornersSSE[0]; }
    inline __m128* maxCornerSSE() { return &cornersSSE[3]; }

#if defined(__aarch64__)
    inline float32x4_t* minCornerNeon() { return &cornersNeon[0]; }
    inline float32x4_t* maxCornerNeon() { return &cornersNeon[3]; }
#endif

    inline void setBBox(int boxNum, const FVec4& minCorner, const FVec4& maxCorner) {
        cornersFloat[0][0][boxNum] = std::min(minCorner.x, maxCorner.x);
        cornersFloat[0][1][boxNum] = std::min(minCorner.y, maxCorner.y);
        cornersFloat[0][2][boxNum] = std::min(minCorner.z, maxCorner.z);
        cornersFloat[1][0][boxNum] = std::max(minCorner.x, maxCorner.x);
        cornersFloat[1][1][boxNum] = std::max(minCorner.y, maxCorner.y);
        cornersFloat[1][2][boxNum] = std::max(minCorner.x, maxCorner.x);
    }

    BBox4(const BBox& a, const BBox& b, const BBox& c, const BBox& d) {
        setBBox(0, a.minCorner(), a.maxCorner());
        setBBox(1, b.minCorner(), b.maxCorner());
        setBBox(2, c.minCorner(), c.maxCorner());
        setBBox(3, d.minCorner(), d.maxCorner());
    }
};

struct Ray {
    FVec4 direction;
    FVec4 origin;
    float tMin;
    float tMax;

    Ray(const FVec4& direction_, const FVec4& origin_, float tMin_, float tMax_)
        : direction(direction_), origin(origin_), tMin(tMin_), tMax(tMax_) {}
};

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

    tMin = std::max(std::max(ray.tMin, (bbox.corners[near.x] - ray.origin.x) * rdir.x),
                    std::max((bbox.corners[near.y] - ray.origin.y) * rdir.y,
                             (bbox.corners[near.z] - ray.origin.z) * rdir.z));
    tMax = std::min(std::min(ray.tMax, (bbox.corners[far.x] - ray.origin.x) * rdir.x),
                    std::min((bbox.corners[far.y] - ray.origin.y) * rdir.y,
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

int main() {
    Ray ray(FVec4(0.0f, 1.0f, 0.0f), FVec4(0.0f, -1.0f, 0.0f), 0.0f, 100.0f);
    BBox bbox0(FVec4(-0.5f, -0.5f, -0.5f), FVec4(0.5f, 0.5f, 0.5f));
    BBox bbox1(FVec4(1.5f, 1.5f, 1.5f), FVec4(2.0f, 2.0f, 2.0f));
    BBox bbox2(FVec4(-2.0f, -2.0f, -2.0f), FVec4(2.0f, 2.0f, 2.0f));
    BBox bbox3(FVec4(-1.5f, -1.5f, -1.5f), FVec4(-2.0f, -2.0f, -2.0f));

    auto printResults = [&](const std::string& testName, int elapsedMicroSec, const IVec4& hits,
                            const FVec4& tMins, const FVec4& tMaxs) {
        std::cout << testName << ": " << elapsedMicroSec << " μs" << std::endl;
        for (size_t i = 0; i < 4; i++) {
            std::cout << "  Box " << i << " hit: " << (hits[i] == 1 ? "true" : "false")
                      << std::endl;
            if (hits[i]) {
                std::cout << "      tMin: "
                          << (tMins[i] > 0.0f ? std::to_string(tMins[i]) : "inside box")
                          << std::endl;
                std::cout << "      tMax: " << tMaxs[i] << std::endl;
            }
        }
        std::cout << std::endl;
    };

    IVec4 hits;
    FVec4 tMins, tMaxs;

    const int numTests = 1000;

    Timer timer;
    for (int i = 0; i < numTests; i++) {
        rayBBoxIntersect4Scalar(ray, bbox0, bbox1, bbox2, bbox3, hits, tMins, tMaxs);
    }
    printResults("Scalar", timer.getElapsedMicoSec(), hits, tMins, tMaxs);

    timer.start();
    for (int i = 0; i < numTests; i++) {
        rayBBoxIntersect4ScalarCompact(ray, bbox0, bbox1, bbox2, bbox3, hits, tMins, tMaxs);
    }
    printResults("Scalar Compact", timer.getElapsedMicoSec(), hits, tMins, tMaxs);

    BBox4 bbox4(bbox0, bbox1, bbox2, bbox3);

    timer.start();
    for (int i = 0; i < numTests; i++) {
        rayBBoxIntersect4SSE(ray, bbox4, hits, tMins, tMaxs);
    }
#if defined(__x86_64__)
    printResults("SSE", timer.getElapsedMicoSec(), hits, tMins, tMaxs);
#elif defined(__aarch64__)
    printResults("SSE (via sse2neon)", timer.getElapsedMicoSec(), hits, tMins, tMaxs);
#endif

#if defined(__aarch64__)
    timer.start();
    for (int i = 0; i < numTests; i++) {
        rayBBoxIntersect4Neon(ray, bbox4, hits, tMins, tMaxs);
    }
    printResults("Neon", timer.getElapsedMicoSec(), hits, tMins, tMaxs);
#endif

    return 0;
}
