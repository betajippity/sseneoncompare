#include "sseneoncompare.hpp"

#include <iostream>
#include <chrono>

struct Timer {
    std::chrono::steady_clock::time_point m_startTime;

    Timer() { m_startTime = std::chrono::steady_clock::now(); }
    void start() { m_startTime = std::chrono::steady_clock::now(); }

    long getElapsedMicroSec() const {
        auto end_time = std::chrono::steady_clock::now();
        return (long)std::min(
            (long)std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_startTime)
                .count(),
            std::numeric_limits<long>::max());
    }
};

int main() {
    Ray ray(FVec4(0.0f, 1.0f, 0.0f), FVec4(0.0f, -1.0f, 0.0f), 0.0f, 100.0f);
    BBox bbox0(FVec4(-0.5f, -0.5f, -0.5f), FVec4(0.5f, 0.5f, 0.5f));
    BBox bbox1(FVec4(1.5f, 1.5f, 1.5f), FVec4(2.0f, 2.0f, 2.0f));
    BBox bbox2(FVec4(-2.0f, -2.0f, -2.0f), FVec4(2.0f, 2.0f, 2.0f));
    BBox bbox3(FVec4(-1.5f, -1.5f, -1.5f), FVec4(-2.0f, -2.0f, -2.0f));
    BBox4 bbox4(bbox0, bbox1, bbox2, bbox3);

    auto printResults = [&](const std::string& testName, long elapsedMicroSec, const IVec4& hits,
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
    printResults("Scalar", timer.getElapsedMicroSec(), hits, tMins, tMaxs);

    timer.start();
    for (int i = 0; i < numTests; i++) {
        rayBBoxIntersect4ScalarCompact(ray, bbox0, bbox1, bbox2, bbox3, hits, tMins, tMaxs);
    }
    printResults("Scalar Compact", timer.getElapsedMicroSec(), hits, tMins, tMaxs);

    timer.start();
    for (int i = 0; i < numTests; i++) {
        rayBBoxIntersect4SSE(ray, bbox4, hits, tMins, tMaxs);
    }
#if defined(__x86_64__)
    printResults("SSE", timer.getElapsedMicroSec(), hits, tMins, tMaxs);
#elif defined(__aarch64__)
    printResults("SSE (via sse2neon)", timer.getElapsedMicroSec(), hits, tMins, tMaxs);
#endif

#if defined(__aarch64__)
    timer.start();
    for (int i = 0; i < numTests; i++) {
        rayBBoxIntersect4Neon(ray, bbox4, hits, tMins, tMaxs);
    }
    printResults("Neon", timer.getElapsedMicroSec(), hits, tMins, tMaxs);
#endif

    timer.start();
    for (int i = 0; i < numTests; i++) {
        rayBBoxIntersect4AutoVectorize(ray, bbox4, hits, tMins, tMaxs);
    }
    printResults("Autovectorize", timer.getElapsedMicroSec(), hits, tMins, tMaxs);

    return 0;
}
