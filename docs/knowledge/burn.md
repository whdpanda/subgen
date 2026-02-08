# Burn Subtitles (ffmpeg)

Hard-sub (burn) an SRT into a video using ffmpeg `subtitles` filter.

- Implemented in `subgen.core.video.burn`
- Tool wrapper: `burn_subtitles`

## Position in PR#4c loop (IMPORTANT)

`burn_subtitles` is **optional** and must run **after** the quality loop finishes:

1) `run_subgen_pipeline`
2) `quality_check_subtitles`
3) `fix_subtitles` → `quality_check_subtitles` → ... (until pass or max passes)
4) **optional**: `burn_subtitles`

Do NOT burn before the loop ends, otherwise you may hard-sub a low-quality SRT.

## Inputs / Outputs contract

### Input
- `video_path`: input video
- `srt_path`: SRT to burn (should be the **final** SRT after loop)
- optional:
  - `force_style`: ASS style string passed to ffmpeg `subtitles=...:force_style=...`
  - audio/video encoding options (if exposed by tool args)

### Output (tool must return real paths)
`burn_subtitles` returns:
- `out_video_path`: the real generated video path
- `artifacts`, `meta` (optional)

Agent/runtime must only print paths coming from tool outputs.
Never guess or hallucinate paths.

## Defaults (PR#4c)

- Default pipeline emit is `zh-only` (Chinese mono SRT), so the recommended SRT for burning is:
  - `primary_path` from `run_subgen_pipeline` (typically `<basename>.zh.srt`)
- If user provides a specific `srt_path`, use that.

## ffmpeg notes

- Uses ffmpeg filter: `-vf subtitles=filename='...':force_style='...'`
- `force_style` is optional; keep defaults unless you need to adjust font/size.
- Audio defaults to AAC 192k (implementation detail; may be configurable by the tool).
