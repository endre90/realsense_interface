# RealSense 3D Pose Estimation

This is an attempt at real-time 6D pose estimation of a known 3D object using an Intel RealSense depth camera. It captures live depth and color data, converts it into a 3D point cloud (the "scene"), and then aligns a pre-loaded 3D CAD model (the "template") to the scene using an Iterative Closest Point (ICP) algorithm. The process is visualized in real-time using the `kiss3d` graphics engine.


## How to Run

1.  **Install SDK:** Install [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense). Install from source and apply kernel pathches, the instructions are good.
2. **Possible Fix:** You might need to disable updating `libcurl` if you encounter:
    ```
    make[4]: *** [CMakeFiles/Makefile2:231: docs/libcurl/CMakeFiles/curl-man.dir/all] Error 2
    make[3]: *** [Makefile:136: all] Error 2
    make[2]: *** [CMakeFiles/libcurl.dir/build.make:87: libcurl/src/libcurl-stamp/libcurl-build] Error 2
    make[1]: *** [CMakeFiles/Makefile2:895: CMakeFiles/libcurl.dir/all] Error 2
    make: *** [Makefile:136: all] Error 2
    ```
    by using the CMake build flag `-DCHECK_FOR_UPDATES=false`.
2.  **Connect Camera:** Ensure a compatible RealSense depth camera is connected.
3.  **Update Model Path:** Change the `MODEL_PATH` constant in `main.rs`.
4.  **Build & Run:** Running in release mode is highly recommended for better performance.
    ```sh
    WINIT_UNIX_BACKEND=x11 cargo run --release
    ```