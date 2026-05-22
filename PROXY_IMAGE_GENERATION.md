# Proxy Image Generation — Concepts

How the `.png` files in `data/work/vlm_3d/proxies/<patent_id>/proxy_<i>.png`
are produced, why they exist, and exactly which call chain creates them.

---

## 1. Why we generate proxy images at all

**Stable-Fast-3D (SF3D)** was trained on photorealistic product photos, not
on patent line-art drawings. If you feed it black-and-white line art
directly, it produces noisy or empty meshes. So we insert an intermediate
"art-to-photo" step:

```
patent line art  ─►  photorealistic proxy image  ─►  SF3D  ─►  .glb mesh
```

The proxy preserves the **geometry** of the patent drawing (object shape,
viewpoint, proportions) but renders it with real-world shading, color, and
texture — i.e. the kind of image SF3D expects on its input.

---

## 2. Where the prompt comes from (the `poses` stage)

Before image generation runs, the upstream `poses` stage already produced a
structured JSON description of the object — the **`vlm_schema`** — using a
VLM parser. Fields look roughly like:

```json
{
  "category": "lamp",
  "parts": ["base", "stem", "shade"],
  "materials": ["matte metal", "fabric"],
  "view": "three-quarter front"
}
```

`vlm_3d/augmentor/prompt_builder.py::schema_to_diffusion_prompt` turns this
into a natural-language string that gets handed to the image model:

> "A matte-metal lamp with a fabric shade, three-quarter front view…"

That's the **text half** of the generation call. The **image half** is the
patent line-art itself.

---

## 3. The Gemini call (image-to-image with text guidance)

Implemented in
`src/patent_pipeline/vlm_3d/augmentor/gemini_augmentor.py::augment_figure_with_gemini`:

```python
client = genai.Client(api_key=GEMINI_API_KEY)
source_img = Image.open(cleaned_figure_path)   # patent line art (PIL Image)

instruction = (
    "You are a photorealistic rendering engine. Given a patent line-art "
    "figure and a text description, produce a photorealistic product photo "
    "of the same object, preserving its geometry, proportions, and "
    "viewpoint. Plain neutral background, soft studio lighting.\n\n"
    f"Description: {diffusion_prompt}"
)

for i in range(n_candidates):
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[instruction, source_img],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
```

Two important details:

- **Multimodal input.** `contents=[instruction, source_img]` sends **both**
  the prompt *and* the patent line-art PIL image in one request. Gemini sees
  the drawing directly and is instructed to preserve its geometry.
- **`response_modalities=["TEXT", "IMAGE"]`.** This is what enables
  image-generating output. Without it the model would only return text.
  This is the "Nano Banana" image-gen surface of Gemini 2.5 Flash.

The model used is `gemini-2.5-flash-image` (no `-preview` suffix — the
preview alias 404s on the v1beta endpoint, which is the bug we hit early
on).

---

## 4. Pulling the image bytes out of the response

The response object carries a list of `candidates`, each with
`content.parts`. Image data arrives as an `inline_data` blob (already
base64-decoded into raw bytes by the SDK) separate from any text parts:

```python
for part in candidate.content.parts or []:
    inline = getattr(part, "inline_data", None)
    if inline is not None and inline.data:
        pil_img = Image.open(io.BytesIO(inline.data))   # bytes -> PIL
        images.append(pil_img)
        break
```

So one API call yields **one** PIL image. The function loops `n_candidates`
times (config: `vlm_3d.num_candidates`, currently `3`) and accumulates a
list of PIL images.

---

## 5. Saving them as `.png`

Back in `src/patent_pipeline/vlm_3d/loop.py::prepare_art3d_job`:

```python
proxy_dir = work_dir / "proxies" / record["patent_id"]
proxy_dir.mkdir(parents=True, exist_ok=True)
for i, img in enumerate(proxy_images):
    path = proxy_dir / f"proxy_{i}.png"
    img.save(path)
    proxy_paths.append(str(path))
```

`PIL.Image.save("…proxy_i.png")` encodes the in-memory image as PNG (PIL
infers the format from the `.png` extension) and writes it to disk. That is
where the `.png` files come from physically — PIL is the encoder; Gemini
just supplies the pixel buffer.

---

## 6. Why three candidates instead of one

Image-generation models are stochastic — each call returns a different
sample. Some drift away from the patent drawing (wrong viewpoint, missing
parts) while others stay faithful. We generate **N=3** and then let Gemini
**itself** score them.

---

## 7. Picking the best candidate

`src/patent_pipeline/vlm_3d/augmentor/proxy_selector.py::select_best_proxy`
makes a second Gemini call with `gemini-2.5-flash` (text-only model), passing:

- the original patent line art,
- all N candidate PNGs,
- the JSON schema,
- a prompt: *"Return ONLY the integer index of the best candidate."*

The integer is parsed (with a regex fallback if Gemini adds prose) and the
matching path becomes `best_proxy_path`. That single image is what gets
handed to SF3D in Phase B.

---

## 8. End-to-end data flow for the PNGs

```
USD0937859-D00000.TIF        (original patent line art, never modified)
        │
        │  Gemini 2.5 Flash Image, 3 samples (stochastic image-to-image)
        ▼
proxies/D0937859/proxy_0.png
proxies/D0937859/proxy_1.png      ◄── Gemini 2.5 Flash selector
proxies/D0937859/proxy_2.png          picks one (e.g. proxy_1.png)
        │
        │  best_proxy_path → SF3D batched subprocess
        ▼
reconstructed_meshes/D0937859.glb
```

---

## 9. Concept summary

- The PNGs are **image-to-image generations** conditioned on both the
  patent line art and a schema-derived text prompt.
- They're produced by a **multimodal LLM (Gemini)**, not a local diffusion
  model. No PyTorch / GPU on the parent side — just HTTPS calls.
- The output is `.png` simply because PIL's `save()` honours the file
  extension. The bytes coming back from Gemini are decoded into a PIL image
  first, then re-encoded as PNG on disk.
- They're **disposable intermediates**: only `best_proxy_path` is used by
  SF3D, but all candidates are retained so the selector's choice is
  auditable.
- Sampling **N=3 + self-selection** is the cheapest way to reject bad
  generations without training a separate reranker.

---

## 10. Knobs you can tweak

| Knob                              | Location                                           | Effect                                       |
|-----------------------------------|----------------------------------------------------|----------------------------------------------|
| `vlm_3d.num_candidates`           | `configs/pipeline.yaml`                            | How many proxies per patent.                 |
| `_IMAGE_GEN_MODEL`                | `vlm_3d/augmentor/gemini_augmentor.py`             | Switch image model (e.g. `gemini-3.1-flash-image-preview` if you want). |
| Instruction text                  | `vlm_3d/augmentor/gemini_augmentor.py`             | Adjust lighting / background / style cues.   |
| `schema_to_diffusion_prompt`      | `vlm_3d/augmentor/prompt_builder.py`               | Change how the structured schema becomes text. |
| `_SELECTOR_MODEL`                 | `vlm_3d/augmentor/proxy_selector.py`               | Use a stronger model to pick the best proxy. |
