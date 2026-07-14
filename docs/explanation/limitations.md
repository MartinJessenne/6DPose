# Pipeline Symmetries & Limitations

> [!IMPORTANT]
> **NEEDS-HUMAN-INPUT**
> This page is a stub. Future maintainers need the original author's input to document the exact known limitations, failure cases, and physical symmetries of the target carts.

Below is the proposed outline and questions to address:

## Proposed Outline
1. **Symmetric Target Objects**: How the physical symmetry of the towing carts (180° rotation around Z-axis) is handled by the dual-hypothesis ICP refinement, and potential issues with other symmetrical axes.
2. **Tubular Geometry Sinks**: Why classical point cloud descriptors (like FPFH) struggle with the thin parallel bars of the cart frame, leading to matching failures.
3. **Sensor and Segmentation Sinks**: Sensitivity of the pose accuracy to YOLO mask boundaries and depth sensor noise/reflections.

---

## Questions for the Author / Maintainers

1. **Dual-Hypothesis ICP**:
   - Are there cases where the dual-hypothesis ICP chooses the incorrect hypothesis?
   - What happens when a cart is symmetric but also has asymmetric components (like hitch bars or wheels)? How does that affect matching?
2. **FPFH descriptor failures**:
   - What voxel sizes and search parameters have you found to work best to prevent RANSAC from matching the vertical bars of the carts incorrectly?
3. **Occlusions & Truncations**:
   - How does the pipeline behave when the cart is only partially visible? Does the YOLO segmentation provide enough coverage for a stable global registration, or does it require a minimum depth point density?
