# Baguettotron SAE Feature Explorer

Interactive map of 4,608 sparse autoencoder features extracted from [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron)

The SAE was trained on residual stream activations at layer 48 (of 80) with a TopK architecture (expansion=8, k=32, d_in=576). Features are projected onto a 2D/3D UMAP and color-coded by activation statistics.

**Live version:** [lyramakesmusic.github.io/bread-slicer](https://lyramakesmusic.github.io/bread-slicer)

## What's here

This is a read-only static export of the full explorer. Everything runs client-side -- no server, no GPU.

What works:
- 2D and 3D UMAP feature maps (Three.js)
- Click any feature to see its top activating examples, nearest neighbors, and token statistics
- Token search (type a word, find which features respond to it)
- WebXR/VR mode
- Dark mode
- Feature labels (4,100+ auto-interpreted by Gemini Flash, 64 human-labeled)

What doesn't:
- Live inference (needs the model on GPU)
- UMAP recompute (needs numpy/sklearn server-side)
- Label saving persists to localStorage only, not back to the dataset

## Running locally

```
cd public
python -m http.server 8000
# open http://localhost:8000
```

Or any static file server. The `features/` directory has 4,608 individual JSON files that get lazy-loaded on click, so you do want an actual HTTP server rather than opening the HTML directly.

## File layout

```
index.html              -- single-page app (~155KB)
data/
  feature_data.json     -- UMAP coordinates + per-feature stats
  labels.json           -- human + autointerp labels with confidence
  neighbors_all.json    -- 10 nearest neighbors per feature (cosine sim)
  top_tokens_all.json   -- top 30 tokens per feature by activation
  search_index.json     -- inverted index for client-side token search
features/
  0.json .. 4607.json   -- per-feature detail (examples with token-level activations)
```


