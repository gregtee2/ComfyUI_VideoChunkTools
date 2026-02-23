# MIT License — Copyright (c) 2026 Greg Tee — see LICENSE file
"""
ComfyUI_VideoChunkTools — Utility nodes for generating long videos
by splitting into overlapping chunks with rolling reference frames.

Designed to solve the "context window reversal" problem where video
generation models (like Wan/FantasyPortrait) revert to the starting
state after ~135 frames because the reference image embedding pulls
the generation back.

WORKFLOW (Rolling Reference for FantasyPortrait):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Load your full driving video (e.g., 300 frames)
2. ExtractVideoChunk(driving, chunk_index=0, chunk_frames=81, overlap=16)
   → driving_chunk_0 (frames 0-80)
3. Run FantasyPortrait pipeline with ORIGINAL portrait as reference
   → generated_chunk_0
4. GetFrameByIndex(generated_chunk_0, index=-1) → last_frame
5. ExtractVideoChunk(driving, chunk_index=1, ...) → driving_chunk_1
6. Run FantasyPortrait pipeline with last_frame as NEW reference
   → generated_chunk_1
7. BlendVideoChunks(generated_chunk_0, generated_chunk_1, overlap=16)
   → seamless output
8. Repeat steps 4-7 for additional chunks

The key insight: by using the LAST generated frame as the reference
for the next chunk, the model's identity stays consistent but the
pose/action can progress forward naturally without reversal.
"""

import math
import torch


class BlendVideoChunks:
    """
    Crossfade-blend two video chunks that share overlapping frames
    at the boundary into a single continuous video.
    
    chunk_a ends where chunk_b begins — they share 'overlap_frames'
    frames at the junction. This node smoothly transitions between them.

    Example:
      chunk_a = 81 frames (frames 0-80 of final video)
      chunk_b = 81 frames (frames 65-145 of final video)
      overlap = 16 frames
      Result = 146 frames: A[0:65] + crossfade[65:81] + B[16:81]
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chunk_a": ("IMAGE", {
                    "tooltip": "First video chunk (the earlier segment)"
                }),
                "chunk_b": ("IMAGE", {
                    "tooltip": "Second video chunk (continues from chunk_a). "
                               "The first 'overlap_frames' of chunk_b overlap "
                               "with the last 'overlap_frames' of chunk_a."
                }),
                "overlap_frames": ("INT", {
                    "default": 16, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Number of frames shared between the two chunks. "
                               "These frames will be crossfaded."
                }),
                "blend_curve": (["linear", "ease_in_out", "sigmoid"], {
                    "default": "ease_in_out",
                    "tooltip": "Blend curve shape: linear = straight ramp, "
                               "ease_in_out = smooth S-curve (3t²-2t³, best for most cases), "
                               "sigmoid = steep S-curve (snappier transition)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("video", "total_frames",)
    FUNCTION = "blend"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = ("Crossfade-blend two overlapping video chunks into a single "
                   "seamless video. Use with ExtractVideoChunk and GetFrameByIndex "
                   "for rolling-reference long video generation.")

    def blend(self, chunk_a, chunk_b, overlap_frames, blend_curve):
        a_len = chunk_a.shape[0]
        b_len = chunk_b.shape[0]

        if overlap_frames >= a_len:
            raise ValueError(
                f"overlap_frames ({overlap_frames}) must be less than "
                f"chunk_a length ({a_len})"
            )
        if overlap_frames >= b_len:
            raise ValueError(
                f"overlap_frames ({overlap_frames}) must be less than "
                f"chunk_b length ({b_len})"
            )

        # Split into non-overlapping + overlapping regions
        a_unique = chunk_a[:a_len - overlap_frames]   # frames only in A
        a_overlap = chunk_a[a_len - overlap_frames:]   # last N frames of A
        b_overlap = chunk_b[:overlap_frames]           # first N frames of B
        b_unique = chunk_b[overlap_frames:]            # frames only in B

        # Build blend weights (0→1 = A→B transition)
        t = torch.linspace(0.0, 1.0, overlap_frames, device=chunk_a.device,
                           dtype=chunk_a.dtype)

        if blend_curve == "ease_in_out":
            # Hermite smoothstep: 3t² - 2t³ (smooth acceleration/deceleration)
            t = t * t * (3.0 - 2.0 * t)
        elif blend_curve == "sigmoid":
            # Steep logistic S-curve (sharper midpoint transition)
            t = torch.sigmoid((t - 0.5) * 12.0)
            t = (t - t[0]) / (t[-1] - t[0])  # renormalize to exact 0-1

        # Reshape for broadcasting: (overlap_frames, 1, 1, 1) over (H, W, C)
        weights = t.view(-1, 1, 1, 1)

        # Crossfade: A*(1-w) + B*w
        blended = a_overlap * (1.0 - weights) + b_overlap * weights

        # Concatenate: [A unique] + [crossfade zone] + [B unique]
        result = torch.cat([a_unique, blended, b_unique], dim=0)

        return (result, result.shape[0],)


class ExtractVideoChunk:
    """
    Extract a specific chunk from a longer video, with overlap support.
    
    Divides a video into chunks of 'chunk_frames' length, where adjacent
    chunks overlap by 'overlap_frames'. Use chunk_index to get each chunk.
    
    Chunk layout (e.g., 300 frames, chunk=81, overlap=16):
      Chunk 0: frames 0-80    (81 frames)
      Chunk 1: frames 65-145  (81 frames, starts 16 before chunk 0 ends)
      Chunk 2: frames 130-210 (81 frames)
      Chunk 3: frames 195-275 (81 frames)
      Chunk 4: frames 219-299 (last chunk, may start closer if near end)
    
    For Wan models: chunk_frames should follow ((n-1)//4)*4+1 rule
    (i.e., 5, 9, 13, ..., 77, 81, 85, ..., 129, 133, 137)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Full driving video / image sequence to chunk"
                }),
                "chunk_index": ("INT", {
                    "default": 0, "min": 0, "max": 1000,
                    "tooltip": "Which chunk to extract (0-based)"
                }),
                "chunk_frames": ("INT", {
                    "default": 81, "min": 2, "max": 2000, "step": 1,
                    "tooltip": "Number of frames per chunk. "
                               "For Wan models, use values like 81, 97, 113, 129 "
                               "(must be 1 mod 4: ((n-1)//4)*4+1)"
                }),
                "overlap_frames": ("INT", {
                    "default": 16, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Number of frames that overlap between adjacent chunks. "
                               "These will be crossfaded by BlendVideoChunks."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN",)
    RETURN_NAMES = ("chunk", "total_chunks", "chunk_index", "is_last_chunk",)
    FUNCTION = "extract"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = ("Extract chunk N from a video for rolling-reference generation. "
                   "Chunks overlap by overlap_frames so they can be blended later.")

    def extract(self, images, chunk_index, chunk_frames, overlap_frames):
        total_frames = images.shape[0]

        if overlap_frames >= chunk_frames:
            raise ValueError(
                f"overlap_frames ({overlap_frames}) must be less than "
                f"chunk_frames ({chunk_frames})"
            )

        stride = chunk_frames - overlap_frames

        # Calculate total number of chunks needed to cover all frames
        if total_frames <= chunk_frames:
            total_chunks = 1
        else:
            total_chunks = 1 + math.ceil((total_frames - chunk_frames) / stride)

        # Calculate start index for this chunk
        start = chunk_index * stride
        end = start + chunk_frames

        # Clamp to video bounds — last chunk may start earlier to stay full-size
        if end > total_frames:
            end = total_frames
            start = max(0, end - chunk_frames)

        is_last = chunk_index >= (total_chunks - 1)

        chunk = images[start:end]

        return (chunk, total_chunks, chunk_index, is_last,)


class GetFrameByIndex:
    """
    Extract a single frame from a video / image batch by index.
    
    Supports negative indexing:
      -1 = last frame (default — use for rolling reference)
      -2 = second-to-last
       0 = first frame
       5 = sixth frame
    
    Primary use case: Get the last frame of a generated chunk
    to use as the reference image for the next chunk.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image batch / video frames"
                }),
                "index": ("INT", {
                    "default": -1, "min": -10000, "max": 10000,
                    "tooltip": "Frame index to extract. Use -1 for last frame "
                               "(ideal for rolling reference), -2 for second-to-last, "
                               "0 for first frame, etc."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frame",)
    FUNCTION = "get_frame"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = ("Get a single frame from a video batch by index. "
                   "Use -1 (last frame) as the reference for the next chunk "
                   "in a rolling-reference workflow.")

    def get_frame(self, images, index):
        total = images.shape[0]

        # Handle negative indexing (Python-style)
        if index < 0:
            index = total + index

        # Clamp to valid range
        index = max(0, min(index, total - 1))

        # Return as single-frame batch (1, H, W, C)
        return (images[index:index + 1],)


class GetFrameRange:
    """
    Extract a range of frames from a video / image batch.
    
    Supports negative indexing for both start and end.
    The end index is EXCLUSIVE (Python slice convention).
    
    Examples:
      start=0, end=16    → first 16 frames
      start=-16, end=0   → last 16 frames (end=0 means "to the end")
      start=10, end=30   → frames 10-29
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image batch / video frames"
                }),
                "start": ("INT", {
                    "default": 0, "min": -10000, "max": 10000,
                    "tooltip": "Start index (inclusive). Negative = from end."
                }),
                "end": ("INT", {
                    "default": 0, "min": -10000, "max": 10000,
                    "tooltip": "End index (exclusive). 0 = to the end of the video. "
                               "Negative = from end."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("frames", "frame_count",)
    FUNCTION = "get_range"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = ("Extract a range of frames from a video batch. "
                   "Supports negative indexing. End=0 means 'to the end'.")

    def get_range(self, images, start, end):
        total = images.shape[0]

        # Handle negative indexing
        if start < 0:
            start = total + start
        start = max(0, min(start, total - 1))

        # end=0 means "to the end"
        if end == 0:
            end = total
        elif end < 0:
            end = total + end
        end = max(start + 1, min(end, total))

        result = images[start:end]
        return (result, result.shape[0],)


class VideoChunkPlanner:
    """
    Calculate the chunking plan for a video — outputs how many chunks
    are needed and the frame ranges, given chunk_frames and overlap.
    
    Useful for planning how many times to run the pipeline and
    understanding the layout before starting generation.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "total_frames": ("INT", {
                    "default": 300, "min": 1, "max": 100000,
                    "tooltip": "Total number of frames in your driving video"
                }),
                "chunk_frames": ("INT", {
                    "default": 81, "min": 2, "max": 2000,
                    "tooltip": "Frames per chunk (e.g., 81 for Wan models)"
                }),
                "overlap_frames": ("INT", {
                    "default": 16, "min": 0, "max": 200,
                    "tooltip": "Frames of overlap between adjacent chunks"
                }),
            },
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("total_chunks", "plan_info",)
    FUNCTION = "plan"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = ("Plan the chunking strategy for long video generation. "
                   "Shows total chunks needed and frame ranges for each.")

    def plan(self, total_frames, chunk_frames, overlap_frames):
        if overlap_frames >= chunk_frames:
            raise ValueError(
                f"overlap_frames ({overlap_frames}) must be less than "
                f"chunk_frames ({chunk_frames})"
            )

        stride = chunk_frames - overlap_frames

        if total_frames <= chunk_frames:
            total_chunks = 1
            final_frames = total_frames
        else:
            total_chunks = 1 + math.ceil((total_frames - chunk_frames) / stride)
            # output video length = total_frames without overlap duplication
            final_frames = chunk_frames + (total_chunks - 1) * stride

        lines = []
        lines.append(f"=== Video Chunk Plan ===")
        lines.append(f"Source video: {total_frames} frames")
        lines.append(f"Chunk size: {chunk_frames} frames")
        lines.append(f"Overlap: {overlap_frames} frames")
        lines.append(f"Stride: {stride} frames")
        lines.append(f"Total chunks needed: {total_chunks}")
        lines.append(f"Output video length: ~{total_frames} frames")
        lines.append(f"")
        lines.append(f"--- Chunk Layout ---")

        for i in range(total_chunks):
            start = i * stride
            end = start + chunk_frames
            if end > total_frames:
                end = total_frames
                start = max(0, end - chunk_frames)

            ref = "original reference" if i == 0 else f"last frame of chunk {i-1}"
            lines.append(
                f"  Chunk {i}: frames {start:>4d}-{end-1:>4d} "
                f"({end - start} frames) | ref: {ref}"
            )

        lines.append(f"")
        lines.append(f"--- Workflow Steps ---")
        lines.append(f"1. Load driving video ({total_frames} frames)")
        for i in range(total_chunks):
            start = i * stride
            end = min(start + chunk_frames, total_frames)
            if end > total_frames:
                start = max(0, total_frames - chunk_frames)
            if i == 0:
                lines.append(f"2. ExtractVideoChunk(index=0) → Generate with original reference")
            else:
                lines.append(
                    f"{2+i}. GetFrameByIndex(chunk_{i-1}, -1) → New reference → "
                    f"ExtractVideoChunk(index={i}) → Generate"
                )
        lines.append(f"{2+total_chunks}. Chain BlendVideoChunks for each pair")

        plan_text = "\n".join(lines)
        return (total_chunks, plan_text,)


class BlendLatentChunks:
    """
    Join two video chunk LATENTS along the temporal dimension BEFORE decoding.

    The overlap_frames parameter is in LATENT temporal space (not pixel frames).
    For Wan models with 4x temporal compression:
        latent_overlap = pixel_overlap / 4
        e.g. 16 pixel frames → 4 latent frames

    Blend modes:
      hard_cut      — No blending. Cuts at the midpoint of the overlap region.
                      Best for rolling-reference workflows where both chunks
                      generated similar content in the overlap zone.
      slerp         — Spherical linear interpolation. Preserves latent vector
                      magnitude (standard for diffusion model interpolation).
                      Avoids the norm-collapse that causes washed-out frames.
      linear        — Simple linear crossfade in latent space.
      ease_in_out   — Hermite smooth-step crossfade.
      sigmoid       — Steep S-curve crossfade.

    Wire BOTH sampler latent outputs into this node, then feed the result
    into a SINGLE VAE decode for the final seamless video.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "overlap_frames": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Overlap in LATENT temporal frames. For Wan: pixel_overlap / 4 (e.g. 16px = 4 latent)"
                }),
                "blend_curve": (["hard_cut", "slerp", "linear", "ease_in_out", "sigmoid"], {
                    "default": "hard_cut",
                    "tooltip": "hard_cut = clean cut at midpoint (best for rolling reference). "
                               "slerp = spherical interpolation (preserves latent magnitude). "
                               "linear/ease_in_out/sigmoid = alpha-blend crossfades."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("latent", "total_latent_frames",)
    FUNCTION = "blend"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = (
        "Join two video chunk latents along temporal dimension before decoding. "
        "hard_cut (recommended): clean cut at overlap midpoint — no dissolve. "
        "slerp: spherical interpolation preserving latent magnitude. "
        "overlap_frames is in LATENT space (Wan: pixel_overlap / 4)."
    )

    @staticmethod
    def _slerp_frame(t, a, b):
        """Spherical linear interpolation between two latent frames.
        Preserves the norm of the vectors — standard for diffusion latent spaces.
        a, b: [B, C, H, W] tensors. t: scalar 0-1."""
        a_f = a.flatten().float()
        b_f = b.flatten().float()
        a_norm = torch.nn.functional.normalize(a_f, dim=0)
        b_norm = torch.nn.functional.normalize(b_f, dim=0)
        dot = (a_norm * b_norm).sum().clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        # Fall back to lerp for nearly-parallel vectors
        if omega.abs() < 1e-5:
            return (a.float() * (1.0 - t) + b.float() * t).to(a.dtype)
        sin_omega = torch.sin(omega)
        wa = torch.sin((1.0 - t) * omega) / sin_omega
        wb = torch.sin(t * omega) / sin_omega
        return (a.float() * wa + b.float() * wb).to(a.dtype)

    def blend(self, latent_a, latent_b, overlap_frames, blend_curve):
        a = latent_a["samples"]
        b = latent_b["samples"]

        if a.dim() != 5 or b.dim() != 5:
            raise ValueError(
                f"Expected 5D video latents [B,C,T,H,W], "
                f"got {a.dim()}D and {b.dim()}D. "
                f"This node is for VIDEO latents, not image latents."
            )

        T_a = a.shape[2]
        T_b = b.shape[2]
        overlap = min(overlap_frames, T_a, T_b)

        # Slice the latent tensors
        a_keep = a[:, :, :T_a - overlap, :, :]
        a_over = a[:, :, T_a - overlap:, :, :]
        b_over = b[:, :, :overlap, :, :]
        b_keep = b[:, :, overlap:, :, :]

        if blend_curve == "hard_cut":
            # Clean cut at the midpoint — no interpolation at all.
            # With rolling reference, both chunks have similar content here,
            # so a clean cut is virtually invisible.
            mid = overlap // 2
            blended = torch.cat([a_over[:, :, :mid, :, :],
                                 b_over[:, :, mid:, :, :]], dim=2)

        elif blend_curve == "slerp":
            # Spherical linear interpolation — preserves latent vector norms.
            # Standard technique for interpolating in diffusion latent spaces.
            # Avoids the washed-out frames caused by lerp norm-collapse.
            frames = []
            for i in range(overlap):
                t = i / max(overlap - 1, 1)
                frame = self._slerp_frame(t, a_over[:, :, i, :, :],
                                              b_over[:, :, i, :, :])
                frames.append(frame.unsqueeze(2))
            blended = torch.cat(frames, dim=2)

        else:
            # Alpha-blend crossfades (linear, ease_in_out, sigmoid)
            weights = torch.linspace(0.0, 1.0, overlap,
                                     device=a.device, dtype=a.dtype)
            if blend_curve == "ease_in_out":
                weights = weights * weights * (3.0 - 2.0 * weights)
            elif blend_curve == "sigmoid":
                weights = torch.sigmoid((weights - 0.5) * 10.0)
                weights = (weights - weights[0]) / (weights[-1] - weights[0])
            weights = weights.reshape(1, 1, -1, 1, 1)
            blended = a_over * (1.0 - weights) + b_over * weights

        # Concatenate: a_unique + overlap_region + b_unique
        result = torch.cat([a_keep, blended, b_keep], dim=2)

        # Preserve any extra keys from latent_a (noise_mask, batch_index, etc.)
        out = {k: v for k, v in latent_a.items() if k != "samples"}
        out["samples"] = result

        return (out, int(result.shape[2]),)


class ConcatVideoChunks:
    """
    Concatenate two decoded video chunks into a single continuous video.

    For rolling-reference workflows where chunk B was generated using chunk A's
    last frame as the CLIP Vision reference:
      - Chunk B's FIRST frame closely matches chunk A's LAST frame
      - Set trim_b_start=1 to drop chunk B's duplicate first frame
      - Result: A[all frames] + B[frame 1 onward] — clean seamless join

    No blending, no overlap, no dissolve — just a clean concatenation.
    The Wan I2V model naturally makes chunk B's first frame match the reference,
    so the boundary is virtually invisible.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chunk_a": ("IMAGE", {
                    "tooltip": "First video chunk (decoded pixels)"
                }),
                "chunk_b": ("IMAGE", {
                    "tooltip": "Second video chunk (decoded pixels). "
                               "Its first frame should match chunk_a's last frame "
                               "if using rolling reference."
                }),
                "trim_b_start": ("INT", {
                    "default": 1, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Number of frames to trim from the START of chunk_b "
                               "before concatenating. Default 1 = drop B's first frame "
                               "(which duplicates A's last frame in rolling-reference workflows)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("video", "total_frames",)
    FUNCTION = "concat"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = (
        "Concatenate two decoded video chunks. Optionally trim B's first N frames "
        "to remove the duplicate reference frame. Simplest join method — no blending needed "
        "when using rolling reference (chunk B's first frame already matches A's last)."
    )

    def concat(self, chunk_a, chunk_b, trim_b_start):
        if trim_b_start > 0:
            chunk_b = chunk_b[trim_b_start:]

        if chunk_b.shape[0] == 0:
            return (chunk_a, chunk_a.shape[0],)

        result = torch.cat([chunk_a, chunk_b], dim=0)
        return (result, result.shape[0],)


# ---------- Registration ----------

NODE_CLASS_MAPPINGS = {
    "BlendVideoChunks": BlendVideoChunks,
    "BlendLatentChunks": BlendLatentChunks,
    "ConcatVideoChunks": ConcatVideoChunks,
    "ExtractVideoChunk": ExtractVideoChunk,
    "GetFrameByIndex": GetFrameByIndex,
    "GetFrameRange": GetFrameRange,
    "VideoChunkPlanner": VideoChunkPlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlendVideoChunks": "Blend Video Chunks (Crossfade)",
    "BlendLatentChunks": "Blend Latent Chunks (Pre-Decode)",
    "ConcatVideoChunks": "Concat Video Chunks",
    "ExtractVideoChunk": "Extract Video Chunk",
    "GetFrameByIndex": "Get Frame By Index",
    "GetFrameRange": "Get Frame Range",
    "VideoChunkPlanner": "Video Chunk Planner",
}
