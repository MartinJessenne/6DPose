# Explanation

These articles provide high-level context, mathematical formulations, and background concepts to help you understand the design decisions and theoretical underpinnings of the 6DPose project.

- [System Architecture](architecture.md): Overview of the dataflow topology, modules, and component interactions.
- [Coordinate Frames & Math](coordinate_frames.md): Mathematical definition of the coordinate frames transform chain (Isaac Sim World to Robot Base Frame).
- [GNC ICP Refinement](gnc_refinement.md): Graduated non-convexity in the SE(2) local refinement — the robust kernel, the per-point stereo noise model that scales it, conditioning, and how the capture radius is derived.
- [Pipeline Symmetries & Failure Modes](limitations.md): Analysis of cart symmetries, occlusion challenges, and known pipeline limitations.
