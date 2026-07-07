Here's the full landscape, organized by family, then filtered through your actual constraints (Orin Nano 8GB, 10+ Hz, CAD available, training OK). One framing note first: every pipeline below shares your existing front-end (YOLO-seg → crop/mask → ROI point cloud), which is a good design and is exactly what most modern pipelines assume. What varies is the pose back-end. Also, "pose relative to base_link" is just a TF chain: all methods give you camera-frame pose; you get base_link via camera extrinsic calibration (hand-eye calibration if the camera is on an arm), so I'll focus on the camera-frame problem.

## 1. Classical geometric methods (training-free, depth-driven)

**PPF + ICP (your current approach).** Drost-style Point Pair Features with Hough-like voting, refined by ICP. Improved variants exist (Vidal et al.'s PPF with visibility context and better sampling won early BOP editions; Halcon's surface-based matching is a polished commercial implementation). Weaknesses: sensitive to depth noise and clutter, struggles with symmetric/repetitive geometry (carts have lots of parallel tubes and planes → many ambiguous point pairs), and typically runs at 100 ms–1 s, so it rarely hits 10 Hz alone.

**Global point-cloud registration.** Extract local 3D descriptors (FPFH) on the ROI cloud and the CAD-sampled cloud, then match + robust fit: RANSAC registration, Fast Global Registration (FGR), or TEASER++ (certifiably robust, handles huge outlier ratios, very fast). Also GO-ICP (branch-and-bound globally optimal ICP) and Super4PCS. TEASER++ + point-to-plane ICP is a strong drop-in upgrade over PPF for your setup.

**ICP-family refinement/tracking.** Point-to-plane ICP, Generalized-ICP, colored ICP. Alone they need a decent initial guess — but that's a feature, not a bug: once you have one good pose, frame-to-frame ICP on the masked cloud is trivially 10+ Hz and can be your steady-state tracker.

**Depth/RGB-D template matching.** LINEMOD-style gradient+normal templates rendered from the CAD over a view sphere, matched to the ROI, then ICP. Old but very fast and still competitive for rigid industrial objects; also what commercial tools (Halcon shape-based 3D matching, MVTec, Zivid/Photoneo SDKs) do.

**Model-driven primitive fitting.** Carts are made of planes, cylinders, and rectangular frames. RANSAC-fit the dominant plane(s)/cylinders in the ROI cloud, match them to known CAD primitives, and solve pose analytically. Extremely fast, interpretable, and robust to partial views — often the most reliable industrial solution when the object has strong structure.

**Ground-plane constrained (3-DoF) estimation.** This one deserves special attention: if the cart is always on the floor, its pose has only 3 unknowns (x, y, yaw) plus a known z/roll/pitch from the floor plane. Detect the floor, project the masked cloud to a bird's-eye 2D footprint, and match the CAD footprint via 2D template matching / chamfer matching / 2D scan-matching (like lidar localization). This collapses the search space by half, kills most symmetry ambiguity, and runs comfortably at high rate on CPU. If your carts never tilt, seriously consider this.

**Fiducials.** AprilTag/ArUco boards on each cart. Not "estimation from geometry," but sub-degree accuracy at hundreds of Hz on CPU. Worth naming because in industrial settings it's often the pragmatic winner if you're allowed to modify the carts.

## 2. Correspondence-based deep learning (train per class on synthetic renders)

All of these train on synthetic images rendered from your CAD (BlenderProc/Isaac Sim with PBR + domain randomization — this is the standard BOP recipe), and most run fast enough for your Jetson after TensorRT export.

**Sparse keypoints + PnP.** Network predicts 2D projections of predefined 3D keypoints; PnP+RANSAC recovers pose. PVNet (pixel-wise voting, robust to occlusion/truncation), BB8, Tekin's YOLO-6D, and notably **YOLO-6D-Pose**, which extends the YOLO architecture you already run into single-stage multi-object pose — you could potentially merge detection+pose into one network and drop a stage. Keypoint+PnP methods are light (tens of ms on Jetson-class hardware), and depth lets you replace PnP with a 3D lift for better accuracy.

**Dense 2D-3D correspondence.** Network predicts, per pixel, the corresponding 3D coordinate on the CAD model (object coordinates / NOCS-style), then PnP-RANSAC: Pix2Pose, DPOD, CDPN, EPOS (handles symmetries explicitly via surface fragments — relevant for carts), **ZebraPose** (hierarchical binary surface codes, among the most accurate per-object methods on BOP), SurfEmb (dense surface embeddings, very accurate but slow — not for you at 10 Hz).

**3D-3D correspondence on the point cloud.** Predict 3D keypoints directly in the masked cloud and solve with least-squares/Kabsch: PVN3D (Hough voting for 3D keypoints), FFB6D (bidirectional RGB-D feature fusion). These exploit your depth channel natively and pair beautifully with your existing masked-cloud front-end.

## 3. Direct regression / holistic deep methods

Network regresses pose (rotation parameterization + translation) directly, sometimes with geometric intermediate features:

- **GDR-Net / GDRNPP** — geometry-guided direct regression; GDRNPP won the BOP challenge and is the de-facto strong baseline for per-object trained pipelines. Fast at inference, TensorRT-friendly.
- PoseCNN, SSD-6D, EfficientPose (single-stage, EfficientDet-based, designed for speed).
- **DenseFusion** and successors (Uni6D, ES6D): per-pixel fusion of RGB and depth embeddings, direct regression + fast iterative refiner. Directly matches your RGB-D + mask input format.

Symmetry warning for this whole family: industrial carts frequently have near-180° yaw symmetry. You must use symmetry-aware losses (ADD-S, or EPOS/SurfEmb-style ambiguity handling) or the network will average between symmetric modes and give garbage.

## 4. Render-and-compare / refinement methods

These take a coarse pose and iteratively refine by comparing renders of the CAD against the image:

- **DeepIM**, **CosyPose** (render&compare refiner; CosyPose also does multi-view consistency if you ever have 2+ cameras), RNNPose, Coupled Iterative Refinement.
- **Differentiable-rendering optimization**: with nvdiffrast/PyTorch3D, directly optimize pose by aligning rendered silhouette+depth against your YOLO mask + measured depth. No training needed, very accurate, a few iterations per frame; a modern replacement for ICP that uses your mask, not just the cloud.
- Classical edge-based refinement (project CAD edges, align to image edges — RAPiD-style), still used in industrial AR tracking.

Refiners aren't standalone pipelines; they're the "ICP slot" of any pipeline above, often better than ICP because they use RGB too.

## 5. Template/retrieval with learned embeddings

- **Augmented Autoencoder (AAE, Sundermeyer)**: renders the CAD over the view sphere, learns an implicit orientation codebook; at runtime, embed the crop and look up rotation, get translation from bbox/depth. Tiny, fast, symmetry-robust by construction (symmetric views collapse in the codebook). Excellent Jetson candidate.
- OVE6D: viewpoint codebook from depth, no per-object training.
- Deep-feature template matching: render templates, match with DINOv2-type features (**FoundPose**), or **GigaPose** (fast novel-object coarse pose from a single correspondence pair). Mostly used as coarse stages for novel-object pipelines.

## 6. Zero-shot / foundation-model pipelines (no per-object training at all)

Given a CAD at inference time only: **MegaPose**, **SAM-6D**, **FoundationPose**, GenFlow, ZeroPose. SAM-6D and FoundationPose are the reference zero-shot RGB-D methods, and pipelines built on them commonly swap the segmentation module for a fine-tuned detector — which is exactly your YOLO-seg, so you'd plug in only their pose module. These are the most accurate and most flexible, but the heaviest. NVIDIA publishes TensorRT FP16 deployments of FoundationPose profiled on Jetson Orin, and users report ~23 FPS for FoundationPose *tracking* on an AGX Orin via Isaac ROS — but the Orin Nano has a fraction of AGX compute and 8 GB shared memory, so full FoundationPose *estimation* at 10 Hz is not realistic there; tracking mode is borderline at best. Treat this family as (a) an offline ground-truth/labeling tool, or (b) a slow init stage.

## 7. Tracking-based pipelines — the key to your 10 Hz requirement

This is the family I'd push hardest for you, because pose *estimation* per frame is wasteful when the cart moves smoothly:

- **Region-based + depth trackers: ICG, ICG+, M3T / SRT3D (Stoiber et al., the 3DObjectTracking library)**. Use the CAD directly, fuse region (silhouette), depth, and texture cues, run at hundreds of Hz *on CPU*. No training. This is arguably the single best fit for an Orin Nano: it barely touches the GPU, leaving it free for YOLO.
- **se(3)-TrackNet**: RGB-D deep tracker trained purely on synthetic renders of your CAD; very fast (90+ Hz on desktop GPUs), robust to occlusion.
- FoundationPose tracking mode (heavier, see above); BundleTrack/BundleSDF (model-free, too heavy).
- Classical: particle-filter depth trackers (dbot), or simply EKF/UKF over per-frame measurements + frame-to-frame ICP.

The canonical robotics architecture is: **slow, accurate initializer (any method from families 1–6, run once or at 0.5–1 Hz for verification/re-init) + fast tracker (family 7) + drift detector**. That's how systems hit 10–100 Hz on edge hardware, and it also smooths your output in base_link (add an EKF on SE(3) there).

## 8. Adjacent families (for completeness)

**Category-level methods** (NOCS, FS-Net, GCE-Pose, etc.) estimate pose+size for unseen instances within a category — relevant only if carts within a class vary geometrically from the CAD. **Multi-view / motion-based**: if the robot moves, fusing poses across viewpoints (CosyPose multi-view, factor-graph fusion) resolves symmetry ambiguities cheaply. **Learned-descriptor registration** (Predator, GeoTransformer, RoITr): deep point-cloud registration between ROI cloud and CAD cloud — accurate but mostly too slow for 10 Hz on your board.

## What I'd actually shortlist for Orin Nano @ 10+ Hz

Realistic per-frame budgets on that board mean: YOLO-seg already costs a meaningful chunk (Jetson Orin Nano YOLO-seg inference typically lands around 25–45 ms), so the pose stage must be light or CPU-based.

1. **ICG/M3T tracking + your PPF (or TEASER++/GDRNPP) as initializer** — minimal training, CPU tracker, likely the fastest path to a robust 10 Hz system.
2. **GDRNPP or ZebraPose trained on BlenderProc renders**, TensorRT FP16, + point-to-plane ICP or differentiable-render refinement — the strongest per-frame estimator if you want stateless estimation.
3. **YOLO-6D-Pose-style unified network** — collapse detection+pose into your existing YOLO stack; least total compute.
4. **Ground-plane 3-DoF matching** — if carts are always floor-bound, this simple pipeline may beat everything on robustness/Hz and deserves a prototype day.
5. FoundationPose/SAM-6D kept offline as an auto-labeling and evaluation oracle for training and benchmarking the above.

Two cross-cutting things to get right regardless of choice: handle the cart's symmetry explicitly (symmetry-aware loss or symmetry-collapsed output space), and validate against BOP-style metrics (ADD-S, VSD, MSSD) on a small real annotated set so you can compare families apples-to-apples.

Want me to go deeper on any branch — e.g., a concrete init+track architecture with ICG/M3T, the synthetic data generation recipe for GDRNPP/ZebraPose, or the ground-plane 3-DoF formulation?