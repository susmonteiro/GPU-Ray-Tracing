# GPU-Ray-Tracing

Ray Tracing Engine for GPUs

This is the final project for the course DD2360 - Applied GPU Programming, at KTH

The final version is the file [raytracing.cu](https://github.com/susmonteiro/GPU-Ray-Tracing/blob/main/raytracing.cu). To compile it and run it, please do as follow:

```
    $ nvcc raytracing.cu -o raytracing

    $ ./raytracing
```

There are some different versions inside the folder `versions`:

- [raytracingPinnedMemory.cu](https://github.com/susmonteiro/GPU-Ray-Tracing/blob/main/versions/raytracingPinnedMemory.cu) uses _pinned memory_ instead of _pageable memory_
- [raytracingUnifiedMemory.cu](https://github.com/susmonteiro/GPU-Ray-Tracing/blob/main/versions/raytracingUnifiedMemory.cu) uses _managed memory_ instead of _pageable memory_
- [raytracingPixels.cu](https://github.com/susmonteiro/GPU-Ray-Tracing/blob/main/versions/raytracingPixels.cu) is an experiment, computing multiple pixels in the same kernel execution
