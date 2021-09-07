typedef float<3> float3;

export void rayBBoxIntersect4ISPC(const uniform float rayDirection[3],
                                  const uniform float rayOrigin[3],
                                  const uniform float rayTMin,
                                  const uniform float rayTMax,
                                  const uniform float bbox4corners[6][4],
                                  uniform float tMins[4],
                                  uniform float tMaxs[4],
                                  uniform int hits[4]) {
    uniform float3 rdir = { 1.0f / rayDirection[0], 1.0f / rayDirection[1],
                            1.0f / rayDirection[2] };

    uniform int near[3] = { 3, 4, 5 };
    if (rdir.x >= 0.0f) {
        near[0] = 0;
    }
    if (rdir.y >= 0.0f) {
        near[1] = 1;
    }
    if (rdir.z >= 0.0f) {
        near[2] = 2;
    }

    uniform int far[3] = { 0, 1, 2 };
    if (rdir.x >= 0.0f) {
        far[0] = 3;
    }
    if (rdir.y >= 0.0f) {
        far[1] = 4;
    }
    if (rdir.z >= 0.0f) {
        far[2] = 5;
    }

    foreach (i = 0...4) {
        tMins[i] = max(max(rayTMin, (bbox4corners[near[0]][i] - rayOrigin[0]) * rdir.x),
                       max((bbox4corners[near[1]][i] - rayOrigin[1]) * rdir.y,
                           (bbox4corners[near[2]][i] - rayOrigin[2]) * rdir.z));
        tMaxs[i] = min(min(rayTMax, (bbox4corners[far[0]][i] - rayOrigin[0]) * rdir.x),
                       min((bbox4corners[far[1]][i] - rayOrigin[1]) * rdir.y,
                           (bbox4corners[far[2]][i] - rayOrigin[2]) * rdir.z));
        hits[i] = tMins[i] <= tMaxs[i];
    }
}
