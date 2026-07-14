# System Architecture

> [!IMPORTANT]
> **NEEDS-HUMAN-INPUT**
> This page is a stub. Future maintainers need the original author's input to document the design rationales behind certain architectural decisions.

Below is the proposed outline and questions to address:

## Proposed Outline
1. **Pipeline Architecture**: Overview of how the YOLO detection front-end feeds cropped frame properties to `MaskedImageFrame` and Open3D point cloud generation.
2. **Component Modularity**: Design decisions separating estimators (`RansacEstimator`, `PPFEstimator`) using the Strategy and Factory patterns.
3. **Dataflow Representation**: How coordinate-system safety is enforced when cropping images and shifting the principal point in camera intrinsics.

---

## Questions for the Author / Maintainers

1. **MaskedImageFrame Design Rationale**:
   - What motivated the creation of the `MaskedImageFrame` data wrapper?
   - Are there specific coordinate alignment bugs (e.g. crop offsets, depth alignment) that this class was explicitly designed to prevent?
2. **Deep Learning Integration**:
   - How do you envision integrating future deep learning estimators (which require raw RGB crops rather than point clouds) into the current interface?
3. **Threading / Async execution**:
   - Is the YOLO-seg to point-cloud pipeline intended to be run in a single-threaded loop, or are there plans to separate detection and registration into async processes for real-time tracking?
