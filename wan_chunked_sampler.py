# MIT License — Copyright (c) 2026 Greg Tee — see LICENSE file
"""
WanChunkedI2VSampler — All-in-one rolling-reference chunked video generation.

Internally handles: encode → sample (two-pass high/low noise) → decode →
extract reference frame → repeat... for as many chunks as needed to produce
`total_frames` of seamless video from a single node.

Replicates the proven v04 concat approach:
  - Each chunk's first frame naturally matches the previous chunk's last frame
    (due to I2V conditioning on the reference image).
  - We drop the overlapping first frame and concatenate — zero artifacts.

Requires: ComfyUI-WanVideoWrapper (kijai)
"""

import math
import importlib
import torch
import gc

from comfy.utils import ProgressBar

# ---------------------------------------------------------------------------
# Lazy import of WanVideoWrapper node classes from ComfyUI's global registry.
# We don't import at module level so that:
#   1. VideoChunkTools can load even if WanVideoWrapper is missing.
#   2. By the time process() runs, all custom nodes are fully registered.
# ---------------------------------------------------------------------------
_wan_cache = {}

def _wan_class(name):
    """Get a WanVideoWrapper node class from ComfyUI's global NODE_CLASS_MAPPINGS."""
    if name in _wan_cache:
        return _wan_cache[name]
    try:
        comfy_nodes = importlib.import_module("nodes")  # ComfyUI root nodes.py
        cls = comfy_nodes.NODE_CLASS_MAPPINGS[name]
        _wan_cache[name] = cls
        return cls
    except (KeyError, AttributeError, ImportError) as exc:
        raise ImportError(
            f"WanChunkedI2VSampler requires ComfyUI-WanVideoWrapper. "
            f"Node class '{name}' not found. Error: {exc}"
        )

# ---------------------------------------------------------------------------
# Scheduler list — dynamically read from WanVideoSampler if available,
# otherwise fall back to a hardcoded subset.
# ---------------------------------------------------------------------------
_FALLBACK_SCHEDULERS = [
    "euler", "euler/beta",
    "unipc", "unipc/beta",
    "dpm++", "dpm++/beta",
    "dpm++_sde", "dpm++_sde/beta",
    "deis",
    "lcm", "lcm/beta",
    "longcat_distill_euler",
    "flowmatch_distill",
    "flowmatch_causvid",
    "res_multistep",
]

_scheduler_list_cache = None

def _get_scheduler_list():
    """Return the scheduler list, preferring WanVideoWrapper's live list."""
    global _scheduler_list_cache
    if _scheduler_list_cache is not None:
        return _scheduler_list_cache
    try:
        comfy_nodes = importlib.import_module("nodes")
        sampler_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("WanVideoSampler")
        if sampler_cls:
            inputs = sampler_cls.INPUT_TYPES()
            _scheduler_list_cache = list(inputs["required"]["scheduler"][0])
            return _scheduler_list_cache
    except Exception:
        pass
    _scheduler_list_cache = list(_FALLBACK_SCHEDULERS)
    return _scheduler_list_cache

# ---------------------------------------------------------------------------
# Image Resizing Helper
# ---------------------------------------------------------------------------
def resize_image_to_target(image, target_width, target_height, keep_proportion, crop_position):
    """
    Resizes an image tensor (B, H, W, C) to target_width and target_height.
    Supports stretch, crop, and pad modes.
    """
    import torch.nn.functional as F
    from comfy.utils import common_upscale

    B, H, W, C = image.shape
    
    if W == target_width and H == target_height:
        return image

    if keep_proportion == "stretch":
        image = image.movedim(-1, 1)
        image = common_upscale(image, target_width, target_height, "bicubic", "disabled")
        image = image.movedim(1, -1)
        return image

    elif keep_proportion == "crop":
        old_aspect = W / H
        new_aspect = target_width / target_height
        if old_aspect > new_aspect:
            crop_w = round(H * new_aspect)
            crop_h = H
        else:
            crop_w = W
            crop_h = round(W / new_aspect)
        
        if crop_position == "center":
            x = (W - crop_w) // 2
            y = (H - crop_h) // 2
        elif crop_position == "top":
            x = (W - crop_w) // 2
            y = 0
        elif crop_position == "bottom":
            x = (W - crop_w) // 2
            y = H - crop_h
        elif crop_position == "left":
            x = 0
            y = (H - crop_h) // 2
        elif crop_position == "right":
            x = W - crop_w
            y = (H - crop_h) // 2
            
        image = image[:, y:y+crop_h, x:x+crop_w, :]
        image = image.movedim(-1, 1)
        image = common_upscale(image, target_width, target_height, "bicubic", "disabled")
        image = image.movedim(1, -1)
        return image

    elif keep_proportion == "pad":
        ratio = min(target_width / W, target_height / H)
        new_width = round(W * ratio)
        new_height = round(H * ratio)
        
        image = image.movedim(-1, 1)
        image = common_upscale(image, new_width, new_height, "bicubic", "disabled")
        image = image.movedim(1, -1)
        
        pad_left = pad_right = pad_top = pad_bottom = 0
        if crop_position == "center":
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left
            pad_top = (target_height - new_height) // 2
            pad_bottom = target_height - new_height - pad_top
        elif crop_position == "top":
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left
            pad_top = 0
            pad_bottom = target_height - new_height
        elif crop_position == "bottom":
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left
            pad_top = target_height - new_height
            pad_bottom = 0
        elif crop_position == "left":
            pad_left = 0
            pad_right = target_width - new_width
            pad_top = (target_height - new_height) // 2
            pad_bottom = target_height - new_height - pad_top
        elif crop_position == "right":
            pad_left = target_width - new_width
            pad_right = 0
            pad_top = (target_height - new_height) // 2
            pad_bottom = target_height - new_height - pad_top
            
        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            image = image.movedim(-1, 1)
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
            image = image.movedim(1, -1)
            
        return image
    
    return image


# ═══════════════════════════════════════════════════════════════════════════
#  WanChunkedI2VSampler
# ═══════════════════════════════════════════════════════════════════════════

class WanChunkedI2VSampler:
    """
    All-in-one chunked video generator for Wan I2V two-pass distilled models.

    Set **total_frames** and **chunk_frames** — the node handles everything:
    encode → sample (high noise pass → low noise pass) → decode →
    extract last frame as reference → repeat for the next chunk.

    Each chunk naturally conditions on the previous chunk's last frame,
    so we just trim the overlapping first frame and concatenate.

    Supports two modes:
      • **Single-pass**: Connect one model. All steps run in one shot.
      • **Two-pass**: Connect a second model to model_b.
        model runs steps 0→split_step, model_b runs split_step→end.
        (Useful for distilled workflows with different LoRA weights per pass.)
    """

    @classmethod
    def INPUT_TYPES(s):
        schedulers = _get_scheduler_list()
        return {
            "required": {
                "model": ("WANVIDEOMODEL", {
                    "tooltip": "Wan I2V model (2.1 or 2.2). Runs all steps in single-pass. In two-pass mode, runs the first steps (0 → split_step)."
                }),
                "vae": ("WANVAE", {
                    "tooltip": "Wan VAE model"
                }),
                "start_image": ("IMAGE", {
                    "tooltip": "Reference image for the first chunk"
                }),
                "total_frames": ("INT", {
                    "default": 161, "min": 5, "max": 9999, "step": 1,
                    "tooltip": "Total frames to generate. Output may be slightly more (rounded to fill the last chunk), then trimmed."
                }),
                "chunk_frames": ("INT", {
                    "default": 81, "min": 5, "max": 241, "step": 4,
                    "tooltip": "Frames per chunk. Auto-normalized to Wan's 4n+1 rule (5, 9, 13, … 77, 81, 85 …)."
                }),
                "width": ("INT", {
                    "default": 832, "min": 128, "max": 2048, "step": 16,
                    "tooltip": "Output video width in pixels."
                }),
                "height": ("INT", {
                    "default": 480, "min": 128, "max": 2048, "step": 16,
                    "tooltip": "Output video height in pixels."
                }),
                "keep_proportion": (["stretch", "crop", "pad"], {
                    "default": "crop",
                    "tooltip": "How to resize the input images to match the target width/height."
                }),
                "crop_position": (["center", "top", "bottom", "left", "right"], {
                    "default": "center",
                    "tooltip": "Where to crop or pad the image."
                }),
                "steps": ("INT", {
                    "default": 8, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Total denoising steps."
                }),
                "cfg": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 30.0, "step": 0.01,
                    "tooltip": "CFG guidance scale. In two-pass mode this applies to the first pass only."
                }),
                "shift": ("FLOAT", {
                    "default": 11.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                    "tooltip": "Scheduler shift parameter."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed. Same seed used for every chunk — variation comes from different reference images."
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload model to CPU after each sampling pass. Highly recommended."
                }),
                "scheduler": (schedulers, {
                    "default": "euler",
                    "tooltip": "Noise scheduler algorithm."
                }),
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable VAE tiling for lower-VRAM decode. May introduce tile seam artifacts."
                }),
            },
            "optional": {
                "model_b": ("WANVIDEOMODEL", {
                    "tooltip": "Optional second model for two-pass sampling. Connect this to split denoising across two models (e.g. different LoRA weights). Leave disconnected for single-pass."
                }),
                "split_step": ("INT", {
                    "default": 4, "min": 1, "max": 99, "step": 1,
                    "tooltip": "Two-pass only: model runs steps 0 → split_step, model_b runs split_step → end. Ignored in single-pass."
                }),
                "cfg_b": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01,
                    "tooltip": "Two-pass only: CFG for the second pass (model_b). Ignored in single-pass."
                }),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", {
                    "tooltip": "Text conditioning from WanVideoTextEncode."
                }),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {
                    "tooltip": "Optional CLIP vision embeddings from WanVideoClipVisionEncode."
                }),
                "feta_args": ("FETAARGS", {
                    "tooltip": "FETA arguments. Recommended: weight=2, start=0, end=1."
                }),
                "context_options": ("WANVIDCONTEXT", {
                    "tooltip": "Context windowing options. In two-pass mode only applied to the first pass (model)."
                }),
                "noise_aug_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Noise augmentation for I2V encoding. 0 = no augmentation."
                }),
                "start_latent_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Strength of the start image in latent space."
                }),
                "end_latent_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "FLF only: Strength of the end image conditioning in latent space. "
                               "Controls how strongly the last chunk is pulled toward the end image. "
                               "Only effective when end_image is connected."
                }),
                "riflex_freq_index": ("INT", {
                    "default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "RIFLEX frequency index. 0 = disabled."
                }),
                "tile_x": ("INT", {
                    "default": 272, "min": 40, "max": 4096, "step": 8,
                    "tooltip": "VAE decode tile width (only when tiling enabled)."
                }),
                "tile_y": ("INT", {
                    "default": 272, "min": 40, "max": 4096, "step": 8,
                    "tooltip": "VAE decode tile height (only when tiling enabled)."
                }),
                "tile_stride_x": ("INT", {
                    "default": 144, "min": 32, "max": 2040, "step": 8,
                    "tooltip": "VAE decode tile stride X (only when tiling enabled)."
                }),
                "tile_stride_y": ("INT", {
                    "default": 128, "min": 32, "max": 2040, "step": 8,
                    "tooltip": "VAE decode tile stride Y (only when tiling enabled)."
                }),
                "end_image": ("IMAGE", {
                    "tooltip": "Optional target image for the last frame — enables FLF (First-Last-Frame) mode. "
                               "If a batch of images is provided, they will be distributed as target keyframes "
                               "across the chunks (e.g., 3 images for a 3-chunk video)."
                }),
                "end_blend_chunks": ("INT", {
                    "default": 0, "min": 0, "max": 40, "step": 1,
                    "tooltip": "FLF only: Number of FRAMES to crossfade at the FLF boundary. "
                               "The last chunk (with end_image) overlaps with the previous chunk "
                               "by this many frames, and a smooth alpha ramp blends between them. "
                               "0 = hard cut (no crossfade). Try 9–17 for smooth transitions "
                               "(~0.5–1 sec at 16fps). The last chunk is automatically enlarged "
                               "to the next valid 4n+1 size to accommodate the overlap."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("video", "total_frames", "chunks_generated",)
    FUNCTION = "process"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = (
        "All-in-one chunked video generator for Wan I2V models.\n\n"
        "Set total_frames and chunk_frames — the node handles everything:\n"
        "encode → sample → decode → extract reference → repeat.\n\n"
        "Supports single-pass (one model) or two-pass (connect model_b).\n\n"
        "FLF (First-Last-Frame): Connect an end_image to make the video "
        "transition toward a target end frame. For multi-chunk videos, "
        "set end_blend_chunks (e.g. 9–17) for a smooth crossfade overlap.\n\n"
        "Uses the proven concat approach: each chunk's first frame naturally "
        "matches the previous chunk's last frame, so we just drop it and concatenate.\n\n"
        "Requires: ComfyUI-WanVideoWrapper"
    )

    def process(
        self,
        model, vae, start_image,
        total_frames, chunk_frames, width, height,
        keep_proportion, crop_position,
        steps, cfg,
        shift, seed, force_offload, scheduler,
        enable_vae_tiling,
        # Optional
        model_b=None,
        split_step=4,
        cfg_b=1.0,
        text_embeds=None,
        clip_embeds=None,
        feta_args=None,
        context_options=None,
        noise_aug_strength=0.0,
        start_latent_strength=1.0,
        end_latent_strength=1.0,
        riflex_freq_index=0,
        tile_x=272, tile_y=272,
        tile_stride_x=144, tile_stride_y=128,
        end_image=None,
        end_blend_chunks=0,
    ):
        # ── Determine mode ──
        two_pass = model_b is not None

        if two_pass and split_step >= steps:
            raise ValueError(
                f"Two-pass mode: split_step ({split_step}) must be less than steps ({steps}). "
                f"High noise runs steps 0→{split_step}, low noise runs {split_step}→{steps}."
            )

        # ── Resize Input Images ──
        start_image = resize_image_to_target(start_image, width, height, keep_proportion, crop_position)
        if end_image is not None:
            end_image = resize_image_to_target(end_image, width, height, keep_proportion, crop_position)

        # ── Import WanVideoWrapper classes (lazy, from global registry) ──
        Encoder = _wan_class("WanVideoImageToVideoEncode")
        Sampler = _wan_class("WanVideoSampler")
        Decoder = _wan_class("WanVideoDecode")

        encoder_inst = Encoder()
        sampler_inst = Sampler()
        decoder_inst = Decoder()

        # ── Normalize chunk_frames to Wan's 4n+1 rule ──
        chunk_frames = ((chunk_frames - 1) // 4) * 4 + 1
        chunk_frames = max(chunk_frames, 5)

        # ── Estimate number of chunks ──
        # First chunk contributes chunk_frames.
        # Each subsequent chunk contributes (chunk_frames - 1) unique frames
        # because its first frame overlaps with the previous chunk's last.
        if total_frames <= chunk_frames:
            num_chunks = 1
        else:
            remaining = total_frames - chunk_frames
            unique_per_extra_chunk = chunk_frames - 1
            num_chunks = 1 + math.ceil(remaining / max(1, unique_per_extra_chunk))

        expected_output = chunk_frames + max(0, num_chunks - 1) * (chunk_frames - 1)

        # ── FLF crossfade: enlarge last chunk for smooth overlap ──
        flf_crossfade = 0
        last_chunk_frames = chunk_frames
        if end_image is not None and end_blend_chunks > 0 and num_chunks > 1:
            flf_crossfade = min(end_blend_chunks, (chunk_frames - 1) // 2)
            needed = chunk_frames + flf_crossfade
            n = (needed - 1 + 3) // 4   # round up to next valid 4n+1
            last_chunk_frames = 4 * n + 1

        mode_label = f"TWO-PASS (split at step {split_step})" if two_pass else "SINGLE-PASS"
        print(f"\n{'=' * 65}")
        print(f"  WanChunkedI2V — Rolling-Reference Chunked Generation")
        print(f"{'=' * 65}")
        print(f"  Mode             : {mode_label}")
        print(f"  Requested frames : {total_frames}")
        print(f"  Chunk size       : {chunk_frames} frames (4n+1)")
        print(f"  Chunks planned   : {num_chunks}")
        print(f"  Expected output  : {expected_output} frames (trimmed to {total_frames})")
        print(f"  Resolution       : {width} × {height}")
        if two_pass:
            print(f"  Steps            : {steps}  (model 0→{split_step}, model_b {split_step}→end)")
            print(f"  CFG              : model={cfg}, model_b={cfg_b}")
        else:
            print(f"  Steps            : {steps}")
            print(f"  CFG              : {cfg}")
        print(f"  Seed             : {seed}")
        print(f"  Scheduler        : {scheduler}")
        print(f"  Force offload    : {force_offload}")
        if end_image is not None:
            B = end_image.shape[0]
            if B > 1:
                print(f"  FLF Mode         : ON (Multi-Keyframe: {B} target images)")
            else:
                print(f"  FLF Mode         : ON (end_image → last chunk only)")
            if flf_crossfade > 0:
                print(f"  FLF Crossfade    : {flf_crossfade} frames overlap")
                print(f"  Last chunk size  : {last_chunk_frames} frames (enlarged for crossfade)")
            else:
                print(f"  FLF Crossfade    : 0 (hard cut at last chunk boundary)")
            print(f"  End Lat Strength : {end_latent_strength}")
        else:
            print(f"  FLF Mode         : OFF")
        print(f"{'=' * 65}\n")

        # ── Progress bar (chunk-level) ──
        pbar = ProgressBar(num_chunks)

        all_frames = []
        total_generated = 0
        reference_image = start_image
        flf_saved_tail = None   # saved frames for FLF crossfade

        for chunk_idx in range(num_chunks):
            # Safety: stop if we already have enough frames
            if total_generated >= total_frames:
                break

            chunk_label = f"Chunk {chunk_idx + 1}/{num_chunks}"
            print(f"\n  ── {chunk_label} ──")

            # ============================================================
            # Step 1 — ENCODE  reference image → image_embeds
            # ============================================================
            is_last_chunk = (chunk_idx == num_chunks - 1)

            # Multi-Keyframe FLF: Map a batch of end_images to the chunks.
            # If 1 image is provided, it applies to the last chunk.
            # If N images are provided, they apply to the last N chunks.
            chunk_end_image = None
            if end_image is not None:
                B = end_image.shape[0]
                target_idx = B - num_chunks + chunk_idx
                if 0 <= target_idx < B:
                    chunk_end_image = end_image[target_idx:target_idx+1]

            flf_this_chunk = (chunk_end_image is not None)
            current_chunk_frames = last_chunk_frames if (is_last_chunk and flf_crossfade > 0) else chunk_frames

            encode_label = "Encoding ref + FLF end target" if flf_this_chunk else "Encoding reference image"
            if flf_this_chunk and is_last_chunk and flf_crossfade > 0:
                encode_label += f" ({current_chunk_frames}f enlarged)"
            print(f"    [1/4] {encode_label} …")

            encode_kwargs = dict(
                width=width,
                height=height,
                num_frames=current_chunk_frames,
                force_offload=force_offload,
                noise_aug_strength=noise_aug_strength,
                start_latent_strength=start_latent_strength,
                end_latent_strength=end_latent_strength,
                start_image=reference_image,
                clip_embeds=clip_embeds,
                vae=vae,
            )
            if flf_this_chunk:
                encode_kwargs["end_image"] = chunk_end_image

            image_embeds = encoder_inst.process(**encode_kwargs)[0]

            # ============================================================
            # Step 2 (+3) — SAMPLE
            # ============================================================
            if two_pass:
                # ── TWO-PASS: model → model_b ──
                print(f"    [2/4] Sampling pass 1 (steps 0 → {split_step}) …")
                high_latent = sampler_inst.process(
                    model=model,
                    image_embeds=image_embeds,
                    steps=steps,
                    cfg=cfg,
                    shift=shift,
                    seed=seed,
                    force_offload=force_offload,
                    scheduler=scheduler,
                    riflex_freq_index=riflex_freq_index,
                    text_embeds=text_embeds,
                    feta_args=feta_args,
                    context_options=context_options,
                    start_step=0,
                    end_step=split_step,
                    add_noise_to_samples=False,
                )[0]

                print(f"    [3/4] Sampling pass 2 (steps {split_step} → end) …")
                final_latent = sampler_inst.process(
                    model=model_b,
                    image_embeds=image_embeds,
                    steps=steps,
                    cfg=cfg_b,
                    shift=shift,
                    seed=seed,
                    force_offload=force_offload,
                    scheduler=scheduler,
                    riflex_freq_index=riflex_freq_index,
                    text_embeds=text_embeds,
                    samples=high_latent,
                    feta_args=feta_args,
                    context_options=None,
                    start_step=split_step,
                    end_step=-1,
                    add_noise_to_samples=False,
                )[0]

                del high_latent
            else:
                # ── SINGLE-PASS: one model, all steps ──
                print(f"    [2/3] Sampling (all {steps} steps) …")
                final_latent = sampler_inst.process(
                    model=model,
                    image_embeds=image_embeds,
                    steps=steps,
                    cfg=cfg,
                    shift=shift,
                    seed=seed,
                    force_offload=force_offload,
                    scheduler=scheduler,
                    riflex_freq_index=riflex_freq_index,
                    text_embeds=text_embeds,
                    feta_args=feta_args,
                    context_options=context_options,
                    add_noise_to_samples=False,
                )[0]

            # ============================================================
            # Step 3/4 — DECODE  latent → pixel frames
            # ============================================================
            decode_label = "4/4" if two_pass else "3/3"
            print(f"    [{decode_label}] VAE decoding …")
            chunk_images = decoder_inst.decode(
                vae=vae,
                samples=final_latent,
                enable_vae_tiling=enable_vae_tiling,
                tile_x=tile_x,
                tile_y=tile_y,
                tile_stride_x=tile_stride_x,
                tile_stride_y=tile_stride_y,
            )[0]  # IMAGE tensor: (T, H, W, C)

            raw_frame_count = chunk_images.shape[0]
            print(f"    Decoded {raw_frame_count} frames")

            # ── Trim overlapping first frame (all chunks after the first) ──
            if chunk_idx > 0:
                chunk_images = chunk_images[1:]
                print(f"    Trimmed overlap → {chunk_images.shape[0]} unique frames")

            # ── FLF crossfade handling ──
            is_penultimate = (chunk_idx == num_chunks - 2)

            if is_penultimate and flf_crossfade > 0 and end_image is not None:
                # Save tail frames for crossfade with upcoming FLF chunk.
                # Set reference to an earlier frame so the FLF chunk
                # regenerates the overlap zone from that point forward.
                cf = min(flf_crossfade, chunk_images.shape[0] - 2)
                flf_saved_tail = chunk_images[-cf:].clone()
                reference_image = chunk_images[-(cf + 1)].unsqueeze(0).clone()

                # Remove tail from this chunk's output (crossfade replaces it)
                chunk_images = chunk_images[:-cf]
                print(f"    FLF: saved {cf} tail frames, ref set {cf+1} frames back")

                all_frames.append(chunk_images)
                total_generated += chunk_images.shape[0]

            elif is_last_chunk and flf_crossfade > 0 and flf_saved_tail is not None:
                # FLF chunk overlaps with saved tail — crossfade them.
                cf = flf_saved_tail.shape[0]
                flf_head = chunk_images[:cf].clone()
                flf_rest = chunk_images[cf:]

                # Ease-in (cubic) alpha ramp: stays near 0 for most frames,
                # then ramps up steeply at the end.  This keeps the crossfade
                # zone looking like the natural chunk with only the last few
                # frames quickly transitioning to FLF content — far less
                # visible "double exposure" than a linear ramp.
                #   Linear:  0.20, 0.40, 0.60, 0.80  (long ghosting)
                #   Cubic:   0.01, 0.06, 0.22, 0.51  (minimal ghosting)
                alphas = torch.linspace(0.0, 1.0, cf + 2)[1:-1]
                alphas = alphas ** 3  # cubic ease-in
                for i in range(cf):
                    a = alphas[i].item()
                    flf_head[i] = (1.0 - a) * flf_saved_tail[i] + a * flf_head[i]

                print(f"    FLF: crossfaded {cf} overlap frames (cubic ease-in)")

                all_frames.append(flf_head)
                all_frames.append(flf_rest)
                total_generated += cf + flf_rest.shape[0]
                del flf_saved_tail, flf_head

            else:
                # Normal chunk — append and extract reference
                all_frames.append(chunk_images)
                total_generated += chunk_images.shape[0]
                if chunk_idx < num_chunks - 1:
                    reference_image = chunk_images[-1:].clone()
                    print(f"    Extracted last frame as next reference")

            # ── Memory cleanup ──
            del image_embeds, final_latent, chunk_images
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(1)
            print(f"    {chunk_label} DONE  (total so far: {total_generated} frames)")

        # ══════════════════════════════════════════════════════════════
        # Concatenate all chunks
        # ══════════════════════════════════════════════════════════════
        video = torch.cat(all_frames, dim=0)

        # Trim to requested length (last chunk may overshoot)
        if video.shape[0] > total_frames:
            video = video[:total_frames]
            print(f"\n  Trimmed {total_generated} → {total_frames} frames")

        final_count = video.shape[0]
        chunks_done = len(all_frames)

        print(f"\n{'=' * 65}")
        print(f"  WanChunkedI2V COMPLETE")
        print(f"  Output       : {final_count} frames")
        print(f"  Chunks used  : {chunks_done}")
        print(f"{'=' * 65}\n")

        return (video, final_count, chunks_done,)


# ═══════════════════════════════════════════════════════════════════════════
#  WanVideoChunkCalculator
# ═══════════════════════════════════════════════════════════════════════════

class WanVideoChunkCalculator:
    """
    Simple utility node to calculate the exact total_frames needed for a 
    specific number of chunks, taking into account the 1-frame overlap.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_chunks": ("INT", {
                    "default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of chunks you want to generate."
                }),
                "chunk_frames": ("INT", {
                    "default": 81, "min": 5, "max": 241, "step": 4,
                    "tooltip": "Frames per chunk. Auto-normalized to Wan's 4n+1 rule."
                }),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_frames",)
    FUNCTION = "calculate"
    CATEGORY = "VideoChunkTools"
    DESCRIPTION = "Calculates the exact total_frames needed for a specific number of chunks."

    def calculate(self, num_chunks, chunk_frames):
        # Normalize to 4n+1
        chunk_frames = ((chunk_frames - 1) // 4) * 4 + 1
        chunk_frames = max(chunk_frames, 5)
        
        if num_chunks <= 1:
            total_frames = chunk_frames
        else:
            # First chunk is full size, subsequent chunks add (chunk_frames - 1)
            total_frames = chunk_frames + (num_chunks - 1) * (chunk_frames - 1)
            
        return (total_frames,)


# ── Registration ──────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WanChunkedI2VSampler": WanChunkedI2VSampler,
    "WanVideoChunkCalculator": WanVideoChunkCalculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanChunkedI2VSampler": "Wan Chunked I2V Sampler ♾️",
    "WanVideoChunkCalculator": "Wan Chunk Calculator 🧮",
}
