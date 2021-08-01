#pragma once
#ifndef SSENEONCOMPARE_HPP
#define SSENEONCOMPARE_HPP

#include <cmath>

#if defined(__x86_64__)
#include <xmmintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#include "sse2neon/sse2neon.h"
#endif

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

inline FVec4 operator/(const float& a, const FVec4& b) {
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
        cornersAlt[0][0] = fmin(minCorner.x, maxCorner.x);
        cornersAlt[0][1] = fmin(minCorner.y, maxCorner.y);
        cornersAlt[0][2] = fmin(minCorner.z, maxCorner.z);
        cornersAlt[1][0] = fmax(minCorner.x, maxCorner.x);
        cornersAlt[1][1] = fmax(minCorner.y, maxCorner.y);
        cornersAlt[1][2] = fmax(minCorner.x, maxCorner.x);
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
        cornersFloat[0][0][boxNum] = fmin(minCorner.x, maxCorner.x);
        cornersFloat[0][1][boxNum] = fmin(minCorner.y, maxCorner.y);
        cornersFloat[0][2][boxNum] = fmin(minCorner.z, maxCorner.z);
        cornersFloat[1][0][boxNum] = fmax(minCorner.x, maxCorner.x);
        cornersFloat[1][1][boxNum] = fmax(minCorner.y, maxCorner.y);
        cornersFloat[1][2][boxNum] = fmax(minCorner.x, maxCorner.x);
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
void rayBBoxIntersect4Scalar(const Ray& ray,
                             const BBox& bbox0,
                             const BBox& bbox1,
                             const BBox& bbox2,
                             const BBox& bbox3,
                             IVec4& hits,
                             FVec4& tMins,
                             FVec4& tMaxs);

/* A much more compact implementation of Williams et al. 2005; this implementation does not
   calculate a negative tMin if the ray origin is inside of the box. */
void rayBBoxIntersect4ScalarCompact(const Ray& ray,
                                    const BBox& bbox0,
                                    const BBox& bbox1,
                                    const BBox& bbox2,
                                    const BBox& bbox3,
                                    IVec4& hits,
                                    FVec4& tMins,
                                    FVec4& tMaxs);

// SSE version of the compact Williams et al. 2005 implementation
void rayBBoxIntersect4SSE(const Ray& ray,
                          const BBox4& bbox4,
                          IVec4& hits,
                          FVec4& tMins,
                          FVec4& tMaxs);

#if defined(__aarch64__)
// Neon version of the compact Williams et al. 2005 implementation
void rayBBoxIntersect4Neon(const Ray& ray,
                           const BBox4& bbox4,
                           IVec4& hits,
                           FVec4& tMins,
                           FVec4& tMaxs);
#endif

// Compact scalar version written to be easily autovectorized
void rayBBoxIntersect4AutoVectorize(const Ray& ray,
                                    const BBox4& bbox4,
                                    IVec4& hits,
                                    FVec4& tMins,
                                    FVec4& tMaxs);

#endif
