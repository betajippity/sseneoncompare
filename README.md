A small SSE Neon comparison
===

This project is a small example comparison of 4-wide ray-bounding-box intersection implemented using two different scalar approaches, using SSE, and using Neon.
The specific ray-bounding-box intersection algorithm implemented is adapted from ["An Efficient and Robust Ray-Box Intersection Algorithm"](https://doi.org/10.1080/2151237X.2005.10129188) by Amy Williams, Steve Barrus, Keith Morley, and Peter Shirley, 2005.

This project is an accompaniment to my blog post, ["Porting Takua Renderer to 64-bit ARM- Part 2"](https://blog.yiningkarlli.com/2021/07/porting-takua-to-arm-pt2.html).
For a thorough walkthrough of the code and explaination for what the code does, please see the blog post.

Dependencies
===

The arm64 build depends on [sse2neon](https://github.com/DLTcollab/sse2neon) for the SSE implementation.
[sse2neon](https://github.com/DLTcollab/sse2neon) is included as a submodule; please clone using the following:

```
git clone --recursive
```

You will need a modern C++ compiler with support for C++17.

Build and Run
===

```
cmake . && make && ./sseneoncompare
```

On macOS 11 and up, using the above will automatically build a Universal Binary with x86_64 and arm64 support.
The x86_64 build includes both scalar implementations and the SSE implementation.
The arm64 build includes both scalar implementations, the SSE implementation (using [sse2neon](https://github.com/DLTcollab/sse2neon)), and the Neon implementation.

When run, the program will run each ray-bounding-box intersection implementation 1000 times and report the total time taken.

Licensing
===

Copyright (c) Yining Karl Li 2021. Licensed under Apache 2.0 (see LICENSE.txt for details).
