"""
模型权重可视化服务
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn


app = FastAPI(title="Nano LLM Weight Visualizer")

# Global checkpoint cache
_checkpoint_cache = None
_checkpoint_path = None
_checkpoint_mtime = None


class WeightInfo(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    numel: int
    min: float
    max: float
    mean: float
    std: float
    cv: float  # coefficient of variation (std/mean)
    sparsity: float  # fraction of near-zero values


class LayerInfo(BaseModel):
    name: str
    weights: list[WeightInfo]


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint with caching, reload if file changed"""
    global _checkpoint_cache, _checkpoint_path, _checkpoint_mtime

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    current_mtime = path.stat().st_mtime

    # Return cached if path and mtime unchanged
    if (_checkpoint_cache is not None and
        _checkpoint_path == checkpoint_path and
        _checkpoint_mtime == current_mtime):
        return _checkpoint_cache

    _checkpoint_cache = torch.load(path, map_location="cpu", weights_only=False)
    _checkpoint_path = checkpoint_path
    _checkpoint_mtime = current_mtime
    return _checkpoint_cache


def compute_weight_stats(name: str, tensor: torch.Tensor) -> WeightInfo:
    """Compute statistics for a weight tensor"""
    tensor_np = tensor.float().numpy()

    # Compute sparsity (values near zero)
    threshold = 1e-6
    near_zero = np.abs(tensor_np) < threshold
    sparsity = float(near_zero.sum() / tensor_np.size)

    # Compute coefficient of variation
    mean_val = float(tensor_np.mean())
    std_val = float(tensor_np.std())
    cv = std_val / abs(mean_val) if mean_val != 0 else float('inf') if std_val > 0 else 0.0

    return WeightInfo(
        name=name,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        numel=tensor.numel(),
        min=float(tensor_np.min()),
        max=float(tensor_np.max()),
        mean=mean_val,
        std=std_val,
        cv=cv,
        sparsity=sparsity,
    )


def get_weight_matrix(tensor: torch.Tensor, max_size: int = 100) -> tuple[np.ndarray, list[int]]:
    """Get downsampled weight matrix for visualization"""
    tensor_np = tensor.float().numpy()

    if tensor_np.ndim == 1:
        # 1D tensor - reshape for visualization
        return tensor_np.reshape(-1, 1), tensor_np.shape

    if tensor_np.ndim > 2:
        # Higher dim - flatten first two dims
        tensor_np = tensor_np.reshape(tensor_np.shape[0], -1)

    original_shape = tensor_np.shape

    # Downsample if too large
    if max(tensor_np.shape) > max_size:
        factor_h = max(1, tensor_np.shape[0] // max_size)
        factor_w = max(1, tensor_np.shape[1] // max_size)

        # Average pooling for downsampling
        h = tensor_np.shape[0] // factor_h
        w = tensor_np.shape[1] // factor_w
        tensor_np = tensor_np[:h * factor_h, :w * factor_w]
        tensor_np = tensor_np.reshape(h, factor_h, w, factor_w).mean(axis=(1, 3))

    return tensor_np, original_shape


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the visualization page"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(content=get_embedded_html())


@app.get("/api/checkpoint/info")
async def get_checkpoint_info(checkpoint_path: str = "checkpoints/nano_llm_last.pth"):
    """Get basic checkpoint info"""
    try:
        ckpt = load_checkpoint(checkpoint_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model_state = ckpt["model_state_dict"]

    # Group weights by layer
    layers: dict[str, list[dict]] = {}
    other_weights: list[dict] = []

    for name, tensor in model_state.items():
        parts = name.split(".")
        layer_name = parts[0] if parts else "other"

        # Check if it's a numbered layer (e.g., layers.0, layers.1)
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            layer_name = f"Layer {parts[1]}"

        stats = compute_weight_stats(name, tensor)

        if layer_name not in layers:
            layers[layer_name] = []
        layers[layer_name].append(stats.model_dump())

    # Convert 0-indexed epoch and batch_idx to 1-indexed for display
    epoch = ckpt.get("epoch")
    batch_idx = ckpt.get("batch_idx")

    return JSONResponse({
        "epoch": (epoch + 1) if epoch is not None else None,
        "batch_idx": (batch_idx + 1) if batch_idx is not None else None,
        "total_params": sum(t.numel() for t in model_state.values()),
        "total_layers": len(set(k.split(".")[0] + "." + k.split(".")[1] if k.startswith("layers.") else k.split(".")[0] for k in model_state.keys())),
        "layers": layers,
        "weight_names": list(model_state.keys()),
    })


@app.get("/api/weight/{weight_name:path}/fft")
async def get_weight_fft(
    weight_name: str,
    checkpoint_path: str = "checkpoints/nano_llm_last.pth",
    num_components: int = 6,
    max_samples: int = 200,
):
    """Get FFT of weight tensor with frequency component waveforms"""
    try:
        ckpt = load_checkpoint(checkpoint_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model_state = ckpt["model_state_dict"]

    # Find the weight
    actual_name = weight_name
    if actual_name not in model_state:
        for k in model_state.keys():
            if k.endswith(weight_name) or k == weight_name:
                actual_name = k
                break

    if actual_name not in model_state:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_name}' not found")

    tensor = model_state[actual_name]
    tensor_np = tensor.float().numpy()

    # Flatten to 1D for FFT decomposition
    original_shape = tensor_np.shape
    flattened = tensor_np.flatten()
    n_total = len(flattened)

    # Compute FFT
    fft_result = np.fft.fft(flattened)
    fft_magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(n_total)

    # Get positive frequencies
    n_half = n_total // 2
    pos_freqs = frequencies[:n_half]
    pos_magnitude = fft_magnitude[:n_half]

    # Find top frequency components by magnitude
    top_indices = np.argsort(pos_magnitude)[::-1][:num_components]

    # Generate waveform for each frequency component
    # Downsample for visualization
    sample_step = max(1, n_total // max_samples)
    x = np.arange(0, n_total, sample_step)

    components = []
    for idx in top_indices:
        freq = frequencies[idx]
        mag = fft_magnitude[idx]
        phase = np.angle(fft_result[idx])

        # Generate the sinusoidal component: A * cos(2π * freq * x + phase)
        # For real signals, we need to combine positive and negative frequency
        if idx == 0:
            # DC component (zero frequency)
            waveform = np.full(len(x), mag / n_total)
        else:
            # Get the conjugate component index
            neg_idx = n_total - idx
            # Amplitude (need to divide by n for proper scaling)
            amp = 2 * mag / n_total
            waveform = amp * np.cos(2 * np.pi * freq * x + phase)

        components.append({
            "frequency": float(freq),
            "magnitude": float(mag),
            "phase": float(phase),
            "normalized_amplitude": float(2 * mag / n_total) if idx > 0 else float(mag / n_total),
            "waveform": waveform.tolist(),
        })

    # Also compute filtered signals (low, mid, high frequency bands)
    n_third = n_half // 3

    # Low frequency component (DC + first 1/3 of frequencies)
    low_fft = np.zeros_like(fft_result)
    low_fft[:n_third] = fft_result[:n_third]
    low_fft[-n_third+1:] = fft_result[-n_third+1:]
    low_signal = np.real(np.fft.ifft(low_fft))

    # Mid frequency component (second 1/3)
    mid_fft = np.zeros_like(fft_result)
    mid_fft[n_third:2*n_third] = fft_result[n_third:2*n_third]
    mid_fft[-2*n_third+1:-n_third+1] = fft_result[-2*n_third+1:-n_third+1]
    mid_signal = np.real(np.fft.ifft(mid_fft))

    # High frequency component (last 1/3)
    high_fft = np.zeros_like(fft_result)
    high_fft[2*n_third:n_half] = fft_result[2*n_third:n_half]
    high_fft[-n_half+1:-2*n_third+1] = fft_result[-n_half+1:-2*n_third+1]
    high_signal = np.real(np.fft.ifft(high_fft))

    # Original signal (downsampled)
    original_downsampled = flattened[::sample_step]

    # Downsample band signals
    low_downsampled = low_signal[::sample_step]
    mid_downsampled = mid_signal[::sample_step]
    high_downsampled = high_signal[::sample_step]

    fft_data = {
        "type": "1d",
        "original_shape": list(original_shape),
        "total_samples": n_total,
        "frequencies": pos_freqs.tolist(),
        "magnitude": pos_magnitude.tolist(),
        "components": components,
        "bands": {
            "original": original_downsampled.tolist(),
            "low_freq": low_downsampled.tolist(),
            "mid_freq": mid_downsampled.tolist(),
            "high_freq": high_downsampled.tolist(),
        },
        "x_axis": x.tolist(),
    }

    return JSONResponse({
        "name": actual_name,
        "fft": fft_data,
    })


@app.get("/api/weight/{weight_name:path}")
async def get_weight_data(
    weight_name: str,
    checkpoint_path: str = "checkpoints/nano_llm_last.pth",
    max_size: int = 100,
    bins: int = 50,
):
    """Get weight data for visualization"""
    try:
        ckpt = load_checkpoint(checkpoint_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model_state = ckpt["model_state_dict"]

    # Find the weight (handle URL encoding)
    actual_name = weight_name
    if actual_name not in model_state:
        # Try to find by endswith match
        for k in model_state.keys():
            if k.endswith(weight_name) or k == weight_name:
                actual_name = k
                break

    if actual_name not in model_state:
        raise HTTPException(status_code=404, detail=f"Weight '{weight_name}' not found")

    tensor = model_state[actual_name]
    matrix, original_shape = get_weight_matrix(tensor, max_size)

    # Compute histogram
    tensor_np = tensor.float().numpy().flatten()
    hist, bin_edges = np.histogram(tensor_np, bins=bins)

    # Compute statistics
    stats = compute_weight_stats(actual_name, tensor)

    return JSONResponse({
        "name": actual_name,
        "shape": list(tensor.shape),
        "original_shape": list(original_shape),
        "data": matrix.tolist(),
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
        "stats": stats.model_dump(),
    })


@app.get("/api/weights/histogram")
async def get_all_histograms(
    checkpoint_path: str = "checkpoints/nano_llm_last.pth",
    bins: int = 50,
):
    """Get histograms for all weights"""
    try:
        ckpt = load_checkpoint(checkpoint_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model_state = ckpt["model_state_dict"]

    histograms = {}
    for name, tensor in model_state.items():
        tensor_np = tensor.float().numpy().flatten()
        hist, bin_edges = np.histogram(tensor_np, bins=bins)
        histograms[name] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "mean": float(tensor_np.mean()),
            "std": float(tensor_np.std()),
        }

    return JSONResponse(histograms)


@app.get("/api/layer/{layer_idx}/attention")
async def get_layer_attention(
    layer_idx: int,
    checkpoint_path: str = "checkpoints/nano_llm_last.pth",
):
    """Get attention weights for a specific layer"""
    try:
        ckpt = load_checkpoint(checkpoint_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model_state = ckpt["model_state_dict"]

    prefix = f"layers.{layer_idx}.attention."
    attention_weights = {}

    for name, tensor in model_state.items():
        if name.startswith(prefix):
            key = name[len(prefix):]
            attention_weights[key] = {
                "shape": list(tensor.shape),
                "stats": compute_weight_stats(name, tensor).model_dump(),
            }

    if not attention_weights:
        raise HTTPException(status_code=404, detail=f"Layer {layer_idx} not found")

    return JSONResponse(attention_weights)


def get_embedded_html() -> str:
    """Return embedded HTML for the visualizer"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nano LLM Weight Visualizer</title>
    <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.0/plotly.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 20px;
            border-bottom: 1px solid #0f3460;
        }
        .header h1 {
            font-size: 24px;
            color: #e94560;
        }
        .header .info {
            margin-top: 10px;
            font-size: 14px;
            color: #888;
        }
        .container {
            display: flex;
            height: calc(100vh - 90px);
        }
        .sidebar {
            width: 300px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            overflow-y: auto;
            padding: 10px;
        }
        .sidebar h3 {
            padding: 10px;
            color: #e94560;
            border-bottom: 1px solid #0f3460;
            position: sticky;
            top: 0;
            background: #16213e;
        }
        .layer-group {
            margin-bottom: 10px;
        }
        .layer-header {
            padding: 8px 10px;
            background: #0f3460;
            cursor: pointer;
            border-radius: 4px;
            margin-bottom: 2px;
        }
        .layer-header:hover {
            background: #1a4a7a;
        }
        .layer-weights {
            padding-left: 10px;
            display: none;
        }
        .layer-weights.active {
            display: block;
        }
        .weight-item {
            padding: 6px 10px;
            cursor: pointer;
            border-radius: 3px;
            font-size: 12px;
            color: #aaa;
        }
        .weight-item:hover {
            background: #0f3460;
            color: #fff;
        }
        .weight-item.selected {
            background: #e94560;
            color: #fff;
        }
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .weight-header {
            padding: 15px 20px;
            background: #0f3460;
            border-bottom: 1px solid #16213e;
        }
        .weight-header h2 {
            font-size: 18px;
            margin-bottom: 5px;
        }
        .weight-stats {
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: #888;
        }
        .weight-stats span {
            color: #4fc3f7;
        }
        .plots-container {
            flex: 1;
            display: flex;
            gap: 10px;
            padding: 10px;
            overflow: hidden;
        }
        .plot-box {
            flex: 1;
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .plot-box h4 {
            padding: 10px 15px;
            background: #0f3460;
            font-size: 14px;
        }
        .plot-wrapper {
            flex: 1;
            padding: 10px;
        }
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #888;
        }
        .histogram-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            padding: 10px;
            overflow-y: auto;
        }
        .histogram-card {
            background: #16213e;
            border-radius: 8px;
            padding: 10px;
        }
        .histogram-card h5 {
            font-size: 11px;
            color: #888;
            margin-bottom: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .view-tabs {
            display: flex;
            background: #0f3460;
            padding: 0;
        }
        .view-tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }
        .view-tab:hover {
            background: #16213e;
        }
        .view-tab.active {
            border-bottom-color: #e94560;
            color: #e94560;
        }
        .view-content {
            flex: 1;
            overflow: hidden;
            display: none;
        }
        .view-content.active {
            display: flex;
            flex-direction: column;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Nano LLM Weight Visualizer</h1>
        <div class="info" id="checkpoint-info">Loading...</div>
    </div>

    <div class="container">
        <div class="sidebar">
            <h3>Model Layers</h3>
            <div id="layer-list">Loading...</div>
        </div>

        <div class="main-content">
            <div class="view-tabs">
                <div class="view-tab active" data-view="detail">Detail View</div>
                <div class="view-tab" data-view="overview">Overview</div>
            </div>

            <div class="view-content active" id="detail-view">
                <div class="weight-header">
                    <h2 id="weight-name">Select a weight to visualize</h2>
                    <div class="weight-stats" id="weight-stats"></div>
                </div>
                <div class="plots-container">
                    <div class="plot-box">
                        <h4>Weight Matrix Heatmap</h4>
                        <div class="plot-wrapper" id="heatmap-plot">
                            <div class="loading">Select a weight from the sidebar</div>
                        </div>
                    </div>
                    <div class="plot-box">
                        <h4>Weight Distribution</h4>
                        <div class="plot-wrapper" id="histogram-plot">
                            <div class="loading">Select a weight from the sidebar</div>
                        </div>
                    </div>
                </div>
                <div class="plots-container" style="border-top: 1px solid #0f3460;">
                    <div class="plot-box" style="flex: 1;">
                        <h4>FFT Magnitude Spectrum</h4>
                        <div class="plot-wrapper" id="fft-spectrum-plot">
                            <div class="loading">Select a weight from the sidebar</div>
                        </div>
                    </div>
                    <div class="plot-box" style="flex: 1;">
                        <h4>Frequency Bands Decomposition</h4>
                        <div class="plot-wrapper" id="fft-bands-plot">
                            <div class="loading">Select a weight from the sidebar</div>
                        </div>
                    </div>
                </div>
                <div class="plots-container" style="border-top: 1px solid #0f3460; max-height: 280px;">
                    <div class="plot-box" style="flex: 1;">
                        <h4>Top Frequency Component Waveforms</h4>
                        <div class="plot-wrapper" id="fft-components-plot">
                            <div class="loading">Select a weight from the sidebar</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="view-content" id="overview-view">
                <div class="histogram-grid" id="overview-grid">
                    <div class="loading">Loading overview...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = '';
        let checkpointInfo = null;
        let selectedWeight = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', async () => {
            await loadCheckpointInfo();
            setupTabs();
        });

        async function loadCheckpointInfo() {
            try {
                const response = await fetch(`${API_BASE}/api/checkpoint/info`);
                checkpointInfo = await response.json();

                const infoHtml = `
                    Epoch: ${checkpointInfo.epoch ?? 'N/A'} |
                    Batch: ${checkpointInfo.batch_idx ?? 'N/A'} |
                    Total Params: ${(checkpointInfo.total_params / 1e6).toFixed(2)}M
                `;
                document.getElementById('checkpoint-info').innerHTML = infoHtml;

                renderLayerList(checkpointInfo.layers);
                await loadOverview();
            } catch (error) {
                document.getElementById('checkpoint-info').textContent = 'Failed to load checkpoint info';
                console.error(error);
            }
        }

        function renderLayerList(layers) {
            const container = document.getElementById('layer-list');
            container.innerHTML = '';

            for (const [layerName, weights] of Object.entries(layers)) {
                const group = document.createElement('div');
                group.className = 'layer-group';

                const header = document.createElement('div');
                header.className = 'layer-header';
                header.textContent = layerName;
                header.onclick = () => {
                    const weightsEl = group.querySelector('.layer-weights');
                    weightsEl.classList.toggle('active');
                };

                const weightsEl = document.createElement('div');
                weightsEl.className = 'layer-weights';

                weights.forEach(w => {
                    const item = document.createElement('div');
                    item.className = 'weight-item';
                    item.textContent = w.name.split('.').slice(-2).join('.');
                    item.title = `${w.name}\\nShape: [${w.shape}]\\nMean: ${w.mean.toFixed(4)}, Std: ${w.std.toFixed(4)}`;
                    item.onclick = (e) => {
                        e.stopPropagation();
                        selectWeight(w.name, item);
                    };
                    weightsEl.appendChild(item);
                });

                group.appendChild(header);
                group.appendChild(weightsEl);
                container.appendChild(group);
            }
        }

        async function selectWeight(name, element) {
            // Update selection UI
            document.querySelectorAll('.weight-item').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            selectedWeight = name;

            // Show weight name
            document.getElementById('weight-name').textContent = name;

            // Show loading
            document.getElementById('heatmap-plot').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('histogram-plot').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('fft-spectrum-plot').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('fft-bands-plot').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('fft-components-plot').innerHTML = '<div class="loading">Loading...</div>';

            try {
                const response = await fetch(`${API_BASE}/api/weight/${encodeURIComponent(name)}?max_size=100`);
                const data = await response.json();

                // Update stats
                const stats = data.stats;
                document.getElementById('weight-stats').innerHTML = `
                    Shape: <span>[${stats.shape.join(', ')}]</span> |
                    Mean: <span>${stats.mean.toFixed(6)}</span> |
                    Std: <span>${stats.std.toFixed(6)}</span> |
                    CV: <span>${stats.cv === Infinity ? '∞' : stats.cv.toFixed(4)}</span> |
                    Min: <span>${stats.min.toFixed(6)}</span> |
                    Max: <span>${stats.max.toFixed(6)}</span> |
                    Sparsity: <span>${(stats.sparsity * 100).toFixed(2)}%</span>
                `;

                renderHeatmap(data);
                renderHistogram(data);

                // Load and render FFT
                loadFFT(name);
            } catch (error) {
                console.error('Failed to load weight:', error);
                document.getElementById('heatmap-plot').innerHTML = '<div class="loading">Failed to load weight</div>';
            }
        }

        async function loadFFT(name) {
            try {
                const response = await fetch(`${API_BASE}/api/weight/${encodeURIComponent(name)}/fft`);
                const data = await response.json();
                renderFFT(data);
            } catch (error) {
                console.error('Failed to load FFT:', error);
                document.getElementById('fft-spectrum-plot').innerHTML = '<div class="loading">Failed to load FFT</div>';
                document.getElementById('fft-bands-plot').innerHTML = '<div class="loading">Failed to load FFT</div>';
                document.getElementById('fft-components-plot').innerHTML = '<div class="loading">Failed to load FFT</div>';
            }
        }

        function renderFFT(data) {
            const fftData = data.fft;

            // 1. Render FFT Magnitude Spectrum
            const spectrumContainer = document.getElementById('fft-spectrum-plot');
            spectrumContainer.innerHTML = '';

            const spectrumData = [{
                x: fftData.frequencies,
                y: fftData.magnitude,
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                line: { color: '#4fc3f7', width: 1 },
            }];

            const spectrumLayout = {
                margin: { l: 50, r: 20, t: 10, b: 40 },
                xaxis: { title: 'Frequency' },
                yaxis: { title: 'Magnitude', type: 'log' },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#888' },
            };

            Plotly.newPlot(spectrumContainer, spectrumData, spectrumLayout, { responsive: true });

            // 2. Render Frequency Bands Decomposition (subplots)
            const bandsContainer = document.getElementById('fft-bands-plot');
            bandsContainer.innerHTML = '';

            const x = fftData.x_axis;
            const bands = fftData.bands;
            const colors = ['#e94560', '#f7b731', '#26de81'];
            const bandNames = ['Low Freq', 'Mid Freq', 'High Freq'];
            const bandKeys = ['low_freq', 'mid_freq', 'high_freq'];

            const bandsData = [];
            for (let i = 0; i < 3; i++) {
                bandsData.push({
                    x: x,
                    y: bands[bandKeys[i]],
                    type: 'scatter',
                    mode: 'lines',
                    name: bandNames[i],
                    line: { color: colors[i], width: 1 },
                    xaxis: 'x',
                    yaxis: `y${i + 1}`,
                });
            }

            const bandsLayout = {
                margin: { l: 40, r: 20, t: 10, b: 40 },
                grid: { rows: 3, columns: 1, pattern: 'independent' },
                xaxis: { title: '', domain: [0, 1] },
                yaxis: { title: 'Low', domain: [0.67, 1], titlefont: { size: 10 } },
                yaxis2: { title: 'Mid', domain: [0.34, 0.66], titlefont: { size: 10 } },
                yaxis3: { title: 'High', domain: [0, 0.33], titlefont: { size: 10 } },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#888', size: 10 },
                showlegend: false,
                height: 200,
            };

            Plotly.newPlot(bandsContainer, bandsData, bandsLayout, { responsive: true });

            // 3. Render Top Frequency Components
            const componentsContainer = document.getElementById('fft-components-plot');
            componentsContainer.innerHTML = '';

            const components = fftData.components;
            const componentsData = [];
            const numRows = Math.min(components.length, 6);

            for (let i = 0; i < numRows; i++) {
                const comp = components[i];
                componentsData.push({
                    x: x,
                    y: comp.waveform,
                    type: 'scatter',
                    mode: 'lines',
                    name: `f=${comp.frequency.toFixed(4)} (A=${comp.normalized_amplitude.toFixed(4)})`,
                    line: { width: 1 },
                    xaxis: 'x',
                    yaxis: `y${i + 1}`,
                });
            }

            const componentsLayout = {
                margin: { l: 40, r: 20, t: 10, b: 40 },
                grid: { rows: numRows, columns: 1, pattern: 'independent' },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#888', size: 10 },
                showlegend: false,
                height: 200,
            };

            // Add axis configs
            for (let i = 0; i < numRows; i++) {
                const domainStart = (numRows - 1 - i) / numRows;
                const domainEnd = (numRows - i) / numRows;
                componentsLayout[`yaxis${i + 1}`] = {
                    domain: [domainStart, domainEnd],
                    title: components[i].frequency.toFixed(4),
                    titlefont: { size: 9 },
                };
            }
            componentsLayout.xaxis = { domain: [0, 1] };

            Plotly.newPlot(componentsContainer, componentsData, componentsLayout, { responsive: true });
        }

        function renderHeatmap(data) {
            const container = document.getElementById('heatmap-plot');
            container.innerHTML = '';

            const plotData = [{
                z: data.data,
                type: 'heatmap',
                colorscale: 'RdBu',
                reversescale: true,
            }];

            const layout = {
                margin: { l: 50, r: 20, t: 20, b: 50 },
                xaxis: { title: 'Column', scaleanchor: 'y' },
                yaxis: { title: 'Row', autorange: 'reversed' },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#888' },
            };

            Plotly.newPlot(container, plotData, layout, { responsive: true });
        }

        function renderHistogram(data) {
            const container = document.getElementById('histogram-plot');
            container.innerHTML = '';

            const histData = [{
                x: data.histogram.bin_edges.slice(0, -1),
                y: data.histogram.counts,
                type: 'bar',
                marker: { color: '#4fc3f7' },
            }];

            const layout = {
                margin: { l: 50, r: 20, t: 20, b: 50 },
                xaxis: { title: 'Value' },
                yaxis: { title: 'Count' },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#888' },
                bargap: 0,
            };

            Plotly.newPlot(container, histData, layout, { responsive: true });
        }

        async function loadOverview() {
            const container = document.getElementById('overview-grid');

            try {
                const response = await fetch(`${API_BASE}/api/weights/histogram`);
                const histograms = await response.json();

                container.innerHTML = '';

                for (const [name, data] of Object.entries(histograms)) {
                    const card = document.createElement('div');
                    card.className = 'histogram-card';

                    const title = document.createElement('h5');
                    title.textContent = name.split('.').slice(-2).join('.');
                    title.title = name;
                    card.appendChild(title);

                    const plotDiv = document.createElement('div');
                    plotDiv.style.height = '120px';
                    card.appendChild(plotDiv);

                    container.appendChild(card);

                    // Render mini histogram
                    const histData = [{
                        x: data.bin_edges.slice(0, -1),
                        y: data.counts,
                        type: 'bar',
                        marker: { color: '#e94560' },
                    }];

                    const layout = {
                        margin: { l: 30, r: 10, t: 10, b: 30 },
                        xaxis: { title: '', showgrid: false },
                        yaxis: { title: '', showgrid: false },
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        font: { color: '#666', size: 10 },
                        bargap: 0,
                    };

                    Plotly.newPlot(plotDiv, histData, layout, {
                        responsive: true,
                        displayModeBar: false
                    });

                    // Add click handler
                    card.style.cursor = 'pointer';
                    card.onclick = () => {
                        // Switch to detail view
                        document.querySelector('.view-tab[data-view="detail"]').click();
                        // Find and click the weight item
                        const weightItems = document.querySelectorAll('.weight-item');
                        weightItems.forEach(item => {
                            if (item.textContent === name.split('.').slice(-2).join('.')) {
                                item.click();
                            }
                        });
                    };
                }
            } catch (error) {
                console.error('Failed to load overview:', error);
                container.innerHTML = '<div class="loading">Failed to load overview</div>';
            }
        }

        function setupTabs() {
            const tabs = document.querySelectorAll('.view-tab');
            tabs.forEach(tab => {
                tab.onclick = () => {
                    tabs.forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');

                    document.querySelectorAll('.view-content').forEach(c => c.classList.remove('active'));
                    document.getElementById(`${tab.dataset.view}-view`).classList.add('active');
                };
            });
        }
    </script>
</body>
</html>'''


def main():
    """Run the visualizer server"""
    import argparse

    parser = argparse.ArgumentParser(description="Nano LLM Weight Visualizer")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--checkpoint", default="checkpoints/nano_llm_last.pth", help="Path to checkpoint file")

    args = parser.parse_args()

    # Preload checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint)
    print("Checkpoint loaded successfully!")

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("Open the URL in your browser to visualize the model weights")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
