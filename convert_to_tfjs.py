"""
Manual TF.js converter - bypasses tensorflowjs package entirely.
Converts a Keras model to TF.js LayersModel format (model.json + .bin shards).
Only requires: tensorflow, numpy
"""
import os
import json
import struct
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

MODEL_PATH = "brain_tumor_model.keras"
OUTPUT_DIR = "hf_space/tfjs_model"
SHARD_SIZE_BYTES = 4 * 1024 * 1024  # 4MB shards

print(f"TF version: {tf.__version__}")
print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"Model loaded. Layers: {len(model.layers)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Collect all weight tensors ──────────────────────────────────────────────
weight_specs = []   # for the model.json manifest
all_bytes = bytearray()

for weight in model.weights:
    arr = weight.numpy().astype(np.float32)
    raw = arr.tobytes()
    spec = {
        "name": weight.name,
        "shape": list(arr.shape),
        "dtype": "float32",
        "byteLength": len(raw),
    }
    weight_specs.append(spec)
    all_bytes.extend(raw)

total_bytes = len(all_bytes)
print(f"Total weight bytes: {total_bytes / 1024 / 1024:.2f} MB")

# ── 2. Write weight binary shards ─────────────────────────────────────────────
num_shards = max(1, (total_bytes + SHARD_SIZE_BYTES - 1) // SHARD_SIZE_BYTES)
shard_files = []

for i in range(num_shards):
    start = i * SHARD_SIZE_BYTES
    end = min(start + SHARD_SIZE_BYTES, total_bytes)
    chunk = bytes(all_bytes[start:end])
    fname = f"group1-shard{i+1}of{num_shards}.bin"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, 'wb') as f:
        f.write(chunk)
    shard_files.append({"name": fname, "sizeBytes": len(chunk)})
    print(f"  Written shard: {fname} ({len(chunk)/1024/1024:.2f} MB)")

# ── 3. Build offset table for the weight manifest ─────────────────────────────
weights_manifest_entries = []
byte_offset = 0
for spec in weight_specs:
    weights_manifest_entries.append({
        "name": spec["name"],
        "shape": spec["shape"],
        "dtype": spec["dtype"],
    })

weights_manifest = [
    {
        "paths": [sf["name"] for sf in shard_files],
        "weights": weights_manifest_entries
    }
]

# ── 4. Build model topology (Keras JSON config) ────────────────────────────────
model_config = json.loads(model.to_json())

# ── 5. Write model.json ─────────────────────────────────────────────────────────
model_json = {
    "format": "layers-model",
    "generatedBy": "manual-keras-to-tfjs",
    "convertedBy": "custom-script",
    "modelTopology": model_config,
    "weightsManifest": weights_manifest
}

model_json_path = os.path.join(OUTPUT_DIR, "model.json")
with open(model_json_path, 'w') as f:
    json.dump(model_json, f)

print(f"\n✅ Done! model.json written ({os.path.getsize(model_json_path)/1024:.1f} KB)")
print(f"\nFiles in {OUTPUT_DIR}/:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"  {fname:50s}  {size/1024/1024:.2f} MB")
