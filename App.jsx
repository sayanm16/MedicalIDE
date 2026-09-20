import { useMemo, useRef, useState } from "react";
import "./App.css";

const DATASETS = {
  MNIST: { name: "MNIST", channels: 1, size: 28, classes: 10, description: "Handwritten digits" },
  "Fashion-MNIST": { name: "Fashion-MNIST", channels: 1, size: 28, classes: 10, description: "Clothing images" },
  "CIFAR-10": { name: "CIFAR-10", channels: 3, size: 32, classes: 10, description: "Colour object images" },
};

const LAYER_LIBRARY = {
  Input: { label: "Input", group: "Input & data", params: {} },
  Normalize: { label: "Normalize", group: "Input & data", params: {} },
  Augmentation: { label: "Augmentation", group: "Input & data", params: {} },
  Reshape: { label: "Reshape", group: "Input & data", params: { channels: 1, size: 28 } },
  Flatten: { label: "Flatten", group: "Input & data", params: {} },
  Conv2d: { label: "Convolution", group: "Feature extraction", params: { filters: 32, kernel: 3, padding: 1, stride: 1 } },
  DepthwiseConv: { label: "Depthwise Conv", group: "Feature extraction", params: { multiplier: 1, kernel: 3, padding: 1 } },
  DilatedConv: { label: "Dilated Conv", group: "Feature extraction", params: { filters: 32, kernel: 3, dilation: 2, padding: 2 } },
  PointwiseConv: { label: "1×1 Convolution", group: "Feature extraction", params: { filters: 32 } },
  TransposedConv: { label: "Transposed Conv", group: "Feature extraction", params: { filters: 32, kernel: 2, stride: 2, padding: 0 } },
  ReLU: { label: "ReLU", group: "Activations", params: {} },
  LeakyReLU: { label: "Leaky ReLU", group: "Activations", params: { slope: 0.1 } },
  GELU: { label: "GELU", group: "Activations", params: {} }, SiLU: { label: "SiLU / Swish", group: "Activations", params: {} }, ELU: { label: "ELU", group: "Activations", params: {} }, SELU: { label: "SELU", group: "Activations", params: {} }, Tanh: { label: "Tanh", group: "Activations", params: {} }, Sigmoid: { label: "Sigmoid", group: "Activations", params: {} }, Softplus: { label: "Softplus", group: "Activations", params: {} },
  MaxPool2d: { label: "Max Pool", group: "Feature extraction", params: { kernel: 2 } },
  AveragePool: { label: "Average Pool", group: "Feature extraction", params: { kernel: 2 } },
  AdaptiveAvgPool: { label: "Adaptive Avg Pool", group: "Feature extraction", params: { size: 1 } },
  GlobalAvgPool: { label: "Global Avg Pool", group: "Feature extraction", params: {} },
  BatchNorm: { label: "BatchNorm", group: "Normalization", params: {} }, LayerNorm: { label: "LayerNorm", group: "Normalization", params: {} }, GroupNorm: { label: "GroupNorm", group: "Normalization", params: { groups: 8 } }, InstanceNorm: { label: "InstanceNorm", group: "Normalization", params: {} },
  Dropout: { label: "Dropout", group: "Regularization", params: { rate: 0.25 } }, Dropout2D: { label: "Dropout2D", group: "Regularization", params: { rate: 0.25 } }, DropPath: { label: "DropPath", group: "Regularization", params: { rate: 0.1 } },
  Dense: { label: "Dense", group: "Classifier", params: { units: 128 } },
  Linear: { label: "Linear", group: "Classifier", params: { units: 128 } }, Bilinear: { label: "Bilinear", group: "Classifier", params: { units: 128 } }, ClassifierHead: { label: "Classifier Head", group: "Classifier", params: { units: 10 } },
  Add: { label: "Add", group: "Connections", params: {} }, Multiply: { label: "Multiply", group: "Connections", params: {} }, Concatenate: { label: "Concatenate", group: "Connections", params: {} }, Residual: { label: "Residual Connection", group: "Connections", params: {} }, Skip: { label: "Skip Connection", group: "Connections", params: {} }, Split: { label: "Split", group: "Connections", params: {} }, Merge: { label: "Merge", group: "Connections", params: {} },
  SelfAttention: { label: "Self Attention", group: "Attention / advanced", params: { heads: 4 } }, MultiHeadAttention: { label: "Multi-Head Attention", group: "Attention / advanced", params: { heads: 8 } }, CrossAttention: { label: "Cross Attention", group: "Attention / advanced", params: { heads: 8 } }, ChannelAttention: { label: "Channel Attention", group: "Attention / advanced", params: {} }, SpatialAttention: { label: "Spatial Attention", group: "Attention / advanced", params: {} }, SEBlock: { label: "SE Block", group: "Attention / advanced", params: { reduction: 16 } }, CBAM: { label: "CBAM", group: "Attention / advanced", params: { reduction: 16 } },
  Softmax: { label: "Softmax", group: "Output", params: {} },
  LogSoftmax: { label: "LogSoftmax", group: "Output", params: {} }, RegressionHead: { label: "Regression Head", group: "Output", params: { units: 1 } }, ClassificationHead: { label: "Classification Head", group: "Output", params: { units: 10 } },
};

const LAYER_GROUPS = Object.entries(LAYER_LIBRARY).reduce(
  (groups, [type, item]) => ({ ...groups, [item.group]: [...(groups[item.group] || []), type] }),
  {},
);

const INITIAL_LAYERS = [
  { id: 1, type: "Conv2d", params: { filters: 32, kernel: 3, padding: 1, stride: 1 } },
  { id: 2, type: "ReLU", params: {} },
  { id: 3, type: "MaxPool2d", params: { kernel: 2 } },
  { id: 4, type: "Flatten", params: {} },
  { id: 5, type: "Linear", params: { units: 128 } },
  { id: 6, type: "ReLU", params: {} },
  { id: 7, type: "Linear", params: { units: 10 } },
];

const isActivation = (type) => ["ReLU", "LeakyReLU", "GELU", "SiLU", "ELU", "SELU", "Tanh", "Sigmoid", "Softplus", "Softmax", "LogSoftmax"].includes(type);
const NORM_2D_TYPES = ["BatchNorm", "InstanceNorm", "GroupNorm"];
const IMAGE_ONLY_REGULARIZERS = ["Dropout2D"];
const BRANCH_TARGETS = ["Add", "Multiply", "Concatenate", "Residual", "Skip", "Split", "Merge", "MultiHeadAttention", "CrossAttention"];
const displayShape = (shape) => shape?.kind === "image" ? `${shape.c} × ${shape.h} × ${shape.w}` : `${shape?.n ?? "?"}`;

function inferArchitecture(layers, dataset, connections = []) {
  let shape = { kind: "image", c: dataset.channels, h: dataset.size, w: dataset.size };
  const result = [];
  const issues = [];

  layers.forEach((layer, index) => {
    const input = { ...shape };
    const connection = connections.find((item) => item.to === layer.id);
    const branch = connection ? result.find((item) => item.id === connection.from)?.output : null;
    let note = "";
    if (["Conv2d", "DilatedConv", "PointwiseConv"].includes(layer.type)) {
      if (shape.kind !== "image") note = "Convolutions need an image-shaped input.";
      else {
        const kernel = layer.type === "PointwiseConv" ? 1 : layer.params.kernel;
        const padding = layer.type === "PointwiseConv" ? 0 : layer.params.padding;
        const dilation = layer.params.dilation || 1;
        const stride = layer.type === "PointwiseConv" ? 1 : (layer.params.stride || 1);
        const span = Math.floor((shape.h + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;
        if (span < 1) note = "Kernel is larger than the current feature map.";
        else shape = { kind: "image", c: layer.params.filters, h: span, w: span };
      }
    } else if (layer.type === "DepthwiseConv") {
      if (shape.kind !== "image") note = "Depthwise convolution needs an image-shaped input.";
      else { const span = Math.floor(shape.h + 2 * layer.params.padding - layer.params.kernel + 1); shape = { kind: "image", c: shape.c * layer.params.multiplier, h: span, w: span }; }
    } else if (layer.type === "TransposedConv") {
      if (shape.kind !== "image") note = "Transposed convolution needs an image-shaped input.";
      else { const span = (shape.h - 1) * layer.params.stride - 2 * layer.params.padding + layer.params.kernel; shape = { kind: "image", c: layer.params.filters, h: span, w: span }; }
    } else if (["MaxPool2d", "AveragePool"].includes(layer.type)) {
      if (shape.kind !== "image") note = "Pooling needs an image-shaped input.";
      else {
        const span = Math.floor(shape.h / layer.params.kernel);
        if (span < 1) note = "Pooling window is larger than the feature map.";
        else shape = { ...shape, h: span, w: span };
      }
    } else if (layer.type === "AdaptiveAvgPool") {
      if (shape.kind !== "image") note = "Adaptive pooling needs image-shaped input.";
      else shape = { ...shape, h: layer.params.size, w: layer.params.size };
    } else if (layer.type === "GlobalAvgPool") {
      if (shape.kind !== "image") note = "Global pooling needs image-shaped input.";
      else shape = { kind: "vector", n: shape.c };
    } else if (layer.type === "Reshape") {
      shape = { kind: "image", c: layer.params.channels, h: layer.params.size, w: layer.params.size };
    } else if (layer.type === "Flatten") {
      if (shape.kind === "image") shape = { kind: "vector", n: shape.c * shape.h * shape.w };
      else note = "This is already a vector; Flatten is not needed.";
    } else if (["Linear", "Dense", "Bilinear", "ClassifierHead", "ClassificationHead", "RegressionHead"].includes(layer.type)) {
      if (shape.kind !== "vector") note = "Add Flatten before the first dense layer.";
      else shape = { kind: "vector", n: layer.params.units };
    } else if (["Softmax", "LogSoftmax"].includes(layer.type) && shape.kind !== "vector") note = "This output activation expects a vector of class scores.";
    else if (["ChannelAttention", "SpatialAttention", "SEBlock", "CBAM"].includes(layer.type)) {
      if (shape.kind !== "image") note = `${LAYER_LIBRARY[layer.type].label} needs an image-shaped feature map.`;
    } else if (["SelfAttention", "MultiHeadAttention", "CrossAttention"].includes(layer.type)) {
      if (layer.type === "CrossAttention" && !branch) note = "Choose a context layer, then attach it here as keys and values.";
      else if (branch && branch.kind !== shape.kind) note = "Attention query and context must have the same tensor rank.";
      else if (branch && shape.kind === "image" && branch.c !== shape.c) note = "Attention query and context must have the same channel count.";
      else if (branch && shape.kind === "vector" && branch.n !== shape.n) note = "Attention query and context must have the same feature count.";
    } else if (["Add", "Multiply", "Concatenate", "Residual", "Skip", "Split", "Merge"].includes(layer.type)) {
      if (!branch) note = "Choose an earlier layer, then attach its branch here.";
      else if (["Add", "Multiply", "Residual", "Merge"].includes(layer.type) && displayShape(branch) !== displayShape(shape)) note = `Branch shape ${displayShape(branch)} must match ${displayShape(shape)}.`;
      else if (layer.type === "Concatenate") {
        if (branch.kind !== shape.kind) note = "Both branches must have the same tensor rank.";
        else if (shape.kind === "image" && (branch.h !== shape.h || branch.w !== shape.w)) note = "Concatenate needs matching spatial dimensions.";
        else if (shape.kind === "image") shape = { ...shape, c: shape.c + branch.c };
        else shape = { kind: "vector", n: shape.n + branch.n };
      }
    } else if (NORM_2D_TYPES.includes(layer.type)) {
      if (shape.kind !== "image") note = `${LAYER_LIBRARY[layer.type].label} expects an image-shaped feature map (before Flatten). Use LayerNorm on vectors instead.`;
      else if (layer.type === "GroupNorm" && shape.c % layer.params.groups !== 0) note = `Channel count (${shape.c}) must be divisible by groups (${layer.params.groups}).`;
    } else if (IMAGE_ONLY_REGULARIZERS.includes(layer.type)) {
      if (shape.kind !== "image") note = `${LAYER_LIBRARY[layer.type].label} expects an image-shaped feature map. Use Dropout on vectors instead.`;
    }

    if (note) issues.push({ index, message: note });
    result.push({ ...layer, input, output: { ...shape }, note, connection, branch });
  });
  return { result, issues };
}

function activationValue(type, x, slope = 0.1) {
  if (type === "ReLU") return Math.max(0, x);
  if (type === "LeakyReLU") return x < 0 ? slope * x : x;
  if (type === "GELU") return x * (0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))));
  if (type === "SiLU") return x / (1 + Math.exp(-x));
  if (type === "ELU") return x < 0 ? Math.exp(x) - 1 : x;
  if (type === "SELU") return 1.0507 * (x < 0 ? 1.6733 * (Math.exp(x) - 1) : x);
  if (type === "Tanh") return Math.tanh(x);
  if (type === "Sigmoid") return 1 / (1 + Math.exp(-x));
  if (type === "Softplus") return Math.log(1 + Math.exp(x));
  if (type === "Softmax") return 1 / (1 + Math.exp(-x));
  return x;
}

function App() {
  const [datasetName, setDatasetName] = useState("MNIST");
  const [layers, setLayers] = useState(INITIAL_LAYERS);
  const [connections, setConnections] = useState([]);
  const [draggedId, setDraggedId] = useState(null);
  const [draggedTool, setDraggedTool] = useState(null);
  const [selectedId, setSelectedId] = useState(2);
  const [training, setTraining] = useState({ optimizer: "AdamW", learningRate: 0.001, batchSize: 128, epochs: 12 });
  const [activationInput, setActivationInput] = useState(0.5);
  const nextId = useRef(8);
  const dataset = DATASETS[datasetName];
  const { result: architecture, issues } = useMemo(() => inferArchitecture(layers, dataset, connections), [layers, dataset, connections]);
  const selected = architecture.find((layer) => layer.id === selectedId) || architecture[0];

  function updateLayer(id, key, value) {
    const parsed = Number(value);
    if (Number.isNaN(parsed)) return;
    setLayers((current) => current.map((layer) => layer.id === id ? { ...layer, params: { ...layer.params, [key]: parsed } } : layer));
  }

  function addLayer(type) {
    const id = nextId.current++;
    const layer = { id, type, params: { ...LAYER_LIBRARY[type].params } };
    setLayers((current) => {
      const selectedIndex = current.findIndex((item) => item.id === selectedId);
      const at = selectedIndex < 0 ? current.length : selectedIndex + 1;
      return [...current.slice(0, at), layer, ...current.slice(at)];
    });
    setSelectedId(id);
  }

  function moveLayer(direction) {
    const index = layers.findIndex((layer) => layer.id === selectedId);
    const target = index + direction;
    if (index < 0 || target < 0 || target >= layers.length) return;
    setLayers((current) => current.map((layer, i) => i === index ? current[target] : i === target ? current[index] : layer));
    pruneInvalidConnections(layers, index, target);
  }

  function removeLayer() {
    if (!selected || layers.length === 1) return;
    const index = layers.findIndex((layer) => layer.id === selected.id);
    setLayers((current) => current.filter((layer) => layer.id !== selected.id));
    setConnections((current) => current.filter((connection) => connection.from !== selected.id && connection.to !== selected.id));
    setSelectedId(layers[Math.max(0, index - 1)]?.id);
  }

  function connectBranch(sourceId) {
    if (!selected || !sourceId) return;
    setConnections((current) => [...current.filter((connection) => connection.to !== selected.id), { from: Number(sourceId), to: selected.id }]);
  }

  function detachBranch() {
    if (!selected) return;
    setConnections((current) => current.filter((connection) => connection.to !== selected.id));
  }

  // Drops any branch connection that no longer points from an earlier layer to a
  // later one, given a hypothetical reorder of `list` swapping positions i and j.
  function pruneInvalidConnections(list, i, j) {
    const reordered = list.map((layer, idx) => idx === i ? list[j] : idx === j ? list[i] : layer);
    const indexOf = new Map(reordered.map((layer, idx) => [layer.id, idx]));
    setConnections((current) => current.filter((connection) => {
      const fromIndex = indexOf.get(connection.from);
      const toIndex = indexOf.get(connection.to);
      return fromIndex !== undefined && toIndex !== undefined && fromIndex < toIndex;
    }));
  }

  const finalLayer = architecture.at(-1);
  const classifierIssue = finalLayer?.output?.kind === "vector" && finalLayer.output.n !== dataset.classes
    ? `Your last layer has ${finalLayer.output.n} outputs, but ${datasetName} needs ${dataset.classes} class scores.` : null;
  const code = makeCode(layers, dataset, connections, training);
  const [editableCode, setEditableCode] = useState("");
  const [codeEdited, setCodeEdited] = useState(false);

  const visibleCode = codeEdited ? editableCode : code;

  function beginLayerDrag(event, id) {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("application/neuralgraph-layer", String(id));
    setDraggedId(id);
    setDraggedTool(null);
  }

  function beginToolDrag(event, type) {
    event.dataTransfer.effectAllowed = "copy";
    event.dataTransfer.setData("application/neuralgraph-tool", type);
    setDraggedTool(type);
    setDraggedId(null);
  }

  function addLayerAt(type, targetId = null, after = true) {
    const id = nextId.current++;
    const layer = { id, type, params: { ...LAYER_LIBRARY[type].params } };
    setLayers((current) => {
      const targetIndex = targetId ? current.findIndex((item) => item.id === targetId) : current.length - 1;
      const at = targetId ? targetIndex + (after ? 1 : 0) : current.length;
      return [...current.slice(0, at), layer, ...current.slice(at)];
    });
    setSelectedId(id);
  }

  function dropLayer(targetId, event) {
    const toolType = event.dataTransfer.getData("application/neuralgraph-tool") || draggedTool;
    const sourceId = Number(event.dataTransfer.getData("application/neuralgraph-layer")) || draggedId;
    const after = event.clientY > event.currentTarget.getBoundingClientRect().top + event.currentTarget.getBoundingClientRect().height / 2;
    if (toolType && LAYER_LIBRARY[toolType]) {
      addLayerAt(toolType, targetId, after);
      setDraggedTool(null);
      return;
    }
    if (!sourceId || sourceId === targetId) return;
    const sourceIndex = layers.findIndex((layer) => layer.id === sourceId);
    const targetIndex = layers.findIndex((layer) => layer.id === targetId);
    if (sourceIndex < 0 || targetIndex < 0) return;
    setLayers((current) => {
      const reordered = [...current];
      const [moved] = reordered.splice(sourceIndex, 1);
      const insertionIndex = targetIndex + (after ? 1 : 0) - (sourceIndex < targetIndex ? 1 : 0);
      reordered.splice(insertionIndex, 0, moved);
      return reordered;
    });
    // A drag-drop can move a layer past several others at once, so re-derive
    // validity from the full new order rather than a single pairwise swap.
    setConnections((current) => {
      const reordered = [...layers];
      const [moved] = reordered.splice(sourceIndex, 1);
      const insertionIndex = targetIndex + (after ? 1 : 0) - (sourceIndex < targetIndex ? 1 : 0);
      reordered.splice(insertionIndex, 0, moved);
      const indexOf = new Map(reordered.map((layer, idx) => [layer.id, idx]));
      return current.filter((connection) => {
        const fromIndex = indexOf.get(connection.from);
        const toIndex = indexOf.get(connection.to);
        return fromIndex !== undefined && toIndex !== undefined && fromIndex < toIndex;
      });
    });
    setSelectedId(sourceId);
    setDraggedId(null);
    setDraggedTool(null);
  }

  return <div className="app-shell">
    <header className="topbar">
      <div><span className="eyebrow">NEURALGRAPH / DESIGN STUDIO</span><h1>Build a network you can reason about.</h1></div>
      <span className="export-badge">Diagram-aware PyTorch export</span>
    </header>

    <main className="workspace">
      <aside className="sidebar">
        <section><p className="panel-label">Dataset</p><select value={datasetName} onChange={(event) => setDatasetName(event.target.value)}>{Object.keys(DATASETS).map((name) => <option key={name}>{name}</option>)}</select>
          <p className="muted">{dataset.description}</p><div className="dataset-stats"><span>INPUT <b>{dataset.channels} × {dataset.size} × {dataset.size}</b></span><span>CLASSES <b>{dataset.classes}</b></span></div></section>
        <section><p className="panel-label">Layer toolbox</p><p className="muted">Click to add after the selected layer, or drag onto a layer to place it before or after.</p>
          {Object.entries(LAYER_GROUPS).map(([group, types]) => <div className="tool-group" key={group}><span>{group}</span><div>{types.map((type) => <button key={type} draggable className="tool-button" onDragStart={(event) => beginToolDrag(event, type)} onDragEnd={() => setDraggedTool(null)} onClick={() => addLayer(type)}>+ {LAYER_LIBRARY[type].label}</button>)}</div></div>)}
        </section>
        <section className="training-card"><p className="panel-label">Training setup</p><label>Optimizer<select value={training.optimizer} onChange={(event) => setTraining((current) => ({ ...current, optimizer: event.target.value }))}><option>AdamW</option><option>SGD</option><option>RMSprop</option></select></label><label>Learning rate<input type="number" min="0" step="0.0001" value={training.learningRate} onChange={(event) => setTraining((current) => ({ ...current, learningRate: Number(event.target.value) }))} /></label><label>Batch size<input type="number" min="1" step="1" value={training.batchSize} onChange={(event) => setTraining((current) => ({ ...current, batchSize: Number(event.target.value) }))} /></label><label>Epochs<input type="number" min="1" step="1" value={training.epochs} onChange={(event) => setTraining((current) => ({ ...current, epochs: Number(event.target.value) }))} /></label></section>
      </aside>

      <section className="canvas">
        <div className="canvas-heading"><div><p className="panel-label">Architecture flow</p><h2>Follow the tensor, layer by layer.</h2></div><span className="node-count">{layers.length} layers</span></div>
        <div className="flow" aria-label="Neural network computation graph" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { const type = event.dataTransfer.getData("application/neuralgraph-tool") || draggedTool; if (type && LAYER_LIBRARY[type]) addLayerAt(type); setDraggedTool(null); }}>
          <div className="io-node input-node"><span>INPUT IMAGE</span><b>{dataset.channels} × {dataset.size} × {dataset.size}</b></div>
          {architecture.map((layer, index) => <div className="flow-step" key={layer.id}><div className="connector"><span>{displayShape(layer.input)}</span><i /></div><button draggable onDragStart={(event) => beginLayerDrag(event, layer.id)} onDragEnd={() => { setDraggedId(null); setDraggedTool(null); }} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); event.stopPropagation(); dropLayer(layer.id, event); }} className={`layer-card ${selected?.id === layer.id ? "selected" : ""} ${draggedId === layer.id ? "dragging" : ""} ${draggedTool ? "drop-target" : ""} ${layer.note ? "invalid" : ""}`} onClick={() => setSelectedId(layer.id)}>
            <span className="layer-index" title="Drag to reorder">⠿ {String(index + 1).padStart(2, "0")}</span><div><strong>{LAYER_LIBRARY[layer.type].label}</strong><small>{parameterDescription(layer)}</small></div><div className="shape"><small>OUTPUT</small><b>{displayShape(layer.output)}</b></div>
          </button>{layer.connection && layer.branch && <p className="branch-link">↗ { ["CrossAttention", "MultiHeadAttention"].includes(layer.type) ? "context from" : "branch from" } layer {architecture.findIndex((item) => item.id === layer.connection.from) + 1}</p>}{layer.note && <p className="layer-error">{layer.note}</p>}</div>)}
          <div className="connector"><span>{displayShape(finalLayer?.output)}</span><i /></div><div className="io-node output-node"><span>PREDICTION</span><b>{dataset.classes} classes</b></div>
        </div>
      </section>

      <aside className="inspector">
        <section><p className="panel-label">Layer inspector</p>{selected ? <><div className="inspector-title"><div><h2>{LAYER_LIBRARY[selected.type].label}</h2><p className="muted">Takes {displayShape(selected.input)} → produces {displayShape(selected.output)}</p></div><div className="layer-actions"><button onClick={() => moveLayer(-1)} title="Move up">↑</button><button onClick={() => moveLayer(1)} title="Move down">↓</button><button onClick={removeLayer} title="Delete">×</button></div></div>{BRANCH_TARGETS.includes(selected.type) && <div className="branch-tools"><label>{["CrossAttention", "MultiHeadAttention"].includes(selected.type) ? "Attention context" : "Connect from layer"}<select value={selected.connection?.from || ""} onChange={(event) => connectBranch(event.target.value)}><option value="">Choose an earlier layer…</option>{architecture.slice(0, architecture.findIndex((layer) => layer.id === selected.id)).map((layer, index) => <option key={layer.id} value={layer.id}>Layer {index + 1}: {LAYER_LIBRARY[layer.type].label} ({displayShape(layer.output)})</option>)}</select></label>{selected.connection && <button onClick={detachBranch}>Detach</button>}</div>}<LayerControls layer={selected} updateLayer={updateLayer} /></> : <p className="muted">Select a layer to inspect it.</p>}</section>
        {selected && isActivation(selected.type) && <ActivationLab type={selected.type} slope={selected.params.slope} input={activationInput} setInput={setActivationInput} output={activationValue(selected.type, activationInput, selected.params.slope)} />}
        <section className="advice"><p className="panel-label">Design checks</p>{issues.length === 0 && !classifierIssue ? <p className="good">✓ Shapes are compatible so far.</p> : null}{issues.map((issue) => <p key={issue.index} className="warning">! Layer {issue.index + 1}: {issue.message}</p>)}{classifierIssue && <p className="warning">! {classifierIssue}</p>}{layers.some((layer) => layer.type === "Softmax") && <p className="tip">Softmax turns class scores into probabilities. With PyTorch’s CrossEntropyLoss, leave it out during training and apply it only for display.</p>}</section>
      </aside>
    </main>
    <section className="code-panel"><div><p className="panel-label">Generated blueprint {codeEdited ? "· edited draft" : ""}</p><h2>PyTorch model</h2></div><div className="code-actions"><button onClick={() => setCodeEdited(false)}>Reset generated</button><button onClick={() => navigator.clipboard?.writeText(visibleCode)}>Copy code</button></div><textarea className="code-editor" aria-label="Editable PyTorch model code" spellCheck="false" value={visibleCode} onChange={(event) => { setEditableCode(event.target.value); setCodeEdited(true); }} /></section>
  </div>;
}

function LayerControls({ layer, updateLayer }) {
  const labels = { filters: "Filters", kernel: "Kernel size", padding: "Padding", dilation: "Dilation", stride: "Stride", multiplier: "Channel multiplier", units: "Output units", rate: "Drop probability", slope: "Negative slope", groups: "Groups", channels: "Channels", size: "Spatial size", heads: "Attention heads", reduction: "Reduction ratio" };
  const fields = Object.entries(layer.params).map(([key, value]) => [key, labels[key] || key, ["rate", "slope"].includes(key) ? 0.01 : 1, value]);
  return fields.length ? <div className="controls">{fields.map(([key, label, step, value]) => <label key={key}>{label}<input type="number" min="0" step={step} value={value} onChange={(event) => updateLayer(layer.id, key, event.target.value)} /></label>)}</div> : <p className="explanation">{isActivation(layer.type) ? "Activation functions add non-linearity: they decide which signals pass forward." : layer.type === "Flatten" ? "Flatten preserves every value while turning feature maps into one vector for dense layers." : ["Add", "Multiply", "Concatenate", "Residual", "Skip", "Split", "Merge"].includes(layer.type) ? "Connections are graph operators: they combine or route tensors between branches. This flow preserves the selected tensor while branch wiring is being prepared." : "This layer has no adjustable parameters."}</p>;
}

function ActivationLab({ type, slope, input, setInput, output }) {
  const points = Array.from({ length: 61 }, (_, index) => { const x = -3 + index / 10; const y = activationValue(type, x, slope); return `${(x + 3) / 6 * 260},${95 - y / 3 * 72}`; }).join(" ");
  return <section className="activation-lab"><p className="panel-label">Activation microscope</p><h2>{LAYER_LIBRARY[type].label}</h2><svg viewBox="0 0 260 160" role="img" aria-label={`${type} activation curve`}><line x1="0" y1="95" x2="260" y2="95" /><line x1="130" y1="15" x2="130" y2="145" /><polyline points={points} /><circle cx={(input + 3) / 6 * 260} cy={95 - output / 3 * 72} r="5" /></svg><label>Try a signal: <b>{input.toFixed(1)}</b><input type="range" min="-3" max="3" step="0.1" value={input} onChange={(event) => setInput(Number(event.target.value))} /></label><div className="activation-result"><span>Input <b>{input.toFixed(2)}</b></span><span>Output <b>{output.toFixed(2)}</b></span></div></section>;
}

function parameterDescription(layer) {
  const p = layer.params;
  if (["Conv2d", "DilatedConv", "TransposedConv"].includes(layer.type)) return `${p.filters} filters · ${p.kernel}×${p.kernel} receptive field${p.stride > 1 ? ` · stride ${p.stride}` : ""}`;
  if (layer.type === "DepthwiseConv") return `per-channel convolution × ${p.multiplier}`;
  if (layer.type === "PointwiseConv") return `${p.filters} filters · channel mixing`;
  if (["Linear", "Dense", "Bilinear", "ClassifierHead", "ClassificationHead", "RegressionHead"].includes(layer.type)) return `${p.units} learned features`;
  if (["MaxPool2d", "AveragePool"].includes(layer.type)) return `${p.kernel}×${p.kernel} downsampling`;
  if (["Dropout", "Dropout2D", "DropPath"].includes(layer.type)) return `${Math.round(p.rate * 100)}% signals masked while training`;
  if (["SelfAttention", "MultiHeadAttention", "CrossAttention"].includes(layer.type)) return `${p.heads} attention heads · query attends to context`;
  if (["ChannelAttention", "SpatialAttention", "SEBlock", "CBAM"].includes(layer.type)) return "reweights important features";
  if (layer.type === "LeakyReLU") return `negative slope ${p.slope}`;
  if (isActivation(layer.type)) return "non-linear signal gate";
  if (["Add", "Multiply", "Concatenate", "Residual", "Skip", "Split", "Merge"].includes(layer.type)) return "graph connection operator";
  if (["BatchNorm", "LayerNorm", "GroupNorm", "InstanceNorm"].includes(layer.type)) return "stabilises activation statistics";
  if (["Flatten", "Reshape"].includes(layer.type)) return "changes tensor layout";
  return "preserves tensor shape";
}

function makeCode(layers, dataset, connections, training) {
  let channels = dataset.channels;
  const definitions = [];
  const forward = [];
  const byTarget = new Map(connections.map((connection) => [connection.to, connection]));
  const activationMap = { ReLU: "ReLU()", LeakyReLU: "LeakyReLU(0.1)", GELU: "GELU()", SiLU: "SiLU()", ELU: "ELU()", SELU: "SELU()", Tanh: "Tanh()", Sigmoid: "Sigmoid()", Softplus: "Softplus()" };
  const hasSelfAttention = layers.some((layer) => layer.type === "SelfAttention" || (layer.type === "MultiHeadAttention" && !byTarget.has(layer.id)));
  const hasCrossAttention = layers.some((layer) => layer.type === "CrossAttention" || (layer.type === "MultiHeadAttention" && byTarget.has(layer.id)));
  const architecture = inferArchitecture(layers, dataset, connections).result;

  layers.forEach((layer, index) => {
    const p = layer.params;
    const name = `layer_${index + 1}`;
    const source = byTarget.get(layer.id);
    const branch = source ? `features[${layers.findIndex((item) => item.id === source.from) + 1}]` : null;
    const attentionChannels = architecture[index]?.input?.kind === "image" ? architecture[index].input.c : 1;
    forward.push(`# ${String(index + 1).padStart(2, "0")} · ${LAYER_LIBRARY[layer.type].label}`);

    if (layer.type === "Conv2d") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${p.filters}, ${p.kernel}, stride=${p.stride ?? 1}, padding=${p.padding})`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (layer.type === "DepthwiseConv") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${channels * p.multiplier}, ${p.kernel}, padding=${p.padding}, groups=${channels})`); forward.push(`x = self.${name}(x)`); channels *= p.multiplier; }
    else if (layer.type === "DilatedConv") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${p.filters}, ${p.kernel}, padding=${p.padding}, dilation=${p.dilation})`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (layer.type === "PointwiseConv") { definitions.push(`self.${name} = nn.Conv2d(${channels}, ${p.filters}, 1)`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (layer.type === "TransposedConv") { definitions.push(`self.${name} = nn.ConvTranspose2d(${channels}, ${p.filters}, ${p.kernel}, stride=${p.stride}, padding=${p.padding})`); forward.push(`x = self.${name}(x)`); channels = p.filters; }
    else if (activationMap[layer.type]) { definitions.push(`self.${name} = nn.${activationMap[layer.type]}`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "MaxPool2d") { definitions.push(`self.${name} = nn.MaxPool2d(${p.kernel})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "AveragePool") { definitions.push(`self.${name} = nn.AvgPool2d(${p.kernel})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "AdaptiveAvgPool") { definitions.push(`self.${name} = nn.AdaptiveAvgPool2d((${p.size}, ${p.size}))`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "GlobalAvgPool") { definitions.push(`self.${name} = nn.AdaptiveAvgPool2d(1)`); forward.push(`x = torch.flatten(self.${name}(x), 1)`); }
    else if (layer.type === "Flatten") forward.push("x = torch.flatten(x, 1)");
    else if (["Linear", "Dense", "Bilinear", "ClassifierHead", "ClassificationHead", "RegressionHead"].includes(layer.type)) { definitions.push(`self.${name} = nn.LazyLinear(${p.units})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "Dropout") { definitions.push(`self.${name} = nn.Dropout(${p.rate})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "Dropout2D") { definitions.push(`self.${name} = nn.Dropout2d(${p.rate})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "DropPath") { definitions.push(`self.${name} = nn.Dropout(${p.rate})  # DropPath approximated as Dropout`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "BatchNorm") { definitions.push(`self.${name} = nn.BatchNorm2d(${channels})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "InstanceNorm") { definitions.push(`self.${name} = nn.InstanceNorm2d(${channels})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "GroupNorm") { definitions.push(`self.${name} = nn.GroupNorm(${p.groups}, ${channels})`); forward.push(`x = self.${name}(x)`); }
    else if (layer.type === "LayerNorm") forward.push("x = F.layer_norm(x, x.shape[1:])");
    else if (layer.type === "Reshape") forward.push(`x = x.reshape(x.shape[0], ${p.channels}, ${p.size}, ${p.size})`);
    else if (["Normalize", "Augmentation", "Input"].includes(layer.type)) forward.push(`# ${LAYER_LIBRARY[layer.type].label} is applied in the dataset transform`);
    else if (["Add", "Residual", "Merge"].includes(layer.type)) forward.push(branch ? `x = x + ${branch}` : "# attach a branch to complete this merge");
    else if (layer.type === "Multiply") forward.push(branch ? `x = x * ${branch}` : "# attach a branch to multiply tensors");
    else if (layer.type === "Concatenate") forward.push(branch ? `x = torch.cat((x, ${branch}), dim=1)` : "# attach a branch to concatenate tensors");
    else if (layer.type === "Skip") forward.push(branch ? `x = ${branch}  # skip route` : "# attach a source for this skip route");
    else if (layer.type === "Split") forward.push("skip = x  # preserve this tensor for a later merge");
    else if (["SelfAttention", "MultiHeadAttention"].includes(layer.type)) {
      if (layer.type === "MultiHeadAttention" && branch) {
        definitions.push(`self.${name} = CrossAttentionBlock(channels=${attentionChannels}, heads=${p.heads})`);
        forward.push(`x = self.${name}(x, ${branch})`);
      } else {
        definitions.push(`self.${name} = SelfAttentionBlock(channels=${attentionChannels}, heads=${p.heads})`);
        forward.push(`x = self.${name}(x)`);
      }
    }
    else if (layer.type === "CrossAttention") {
      definitions.push(`self.${name} = CrossAttentionBlock(channels=${attentionChannels}, heads=${p.heads})`);
      forward.push(branch ? `x = self.${name}(x, ${branch})` : "# attach an earlier context layer before running cross attention");
    }
    else if (["ChannelAttention", "SpatialAttention", "SEBlock", "CBAM"].includes(layer.type)) forward.push(`# ${LAYER_LIBRARY[layer.type].label} needs a task-specific image attention block`);
    else if (layer.type === "Softmax") forward.push("x = torch.softmax(x, dim=1)");
    else if (layer.type === "LogSoftmax") forward.push("x = torch.log_softmax(x, dim=1)");
    forward.push(`features[${index + 1}] = x`);
  });

  const optimizerCall = training.optimizer === "AdamW"
    ? `torch.optim.AdamW(\n    model.parameters(),\n    lr=${training.learningRate},\n    weight_decay=1e-4\n)`
    : training.optimizer === "SGD"
      ? `torch.optim.SGD(\n    model.parameters(),\n    lr=${training.learningRate},\n    momentum=0.9\n)`
      : `torch.optim.RMSprop(\n    model.parameters(),\n    lr=${training.learningRate}\n)`;

    return `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

${hasSelfAttention || hasCrossAttention ? `def _valid_heads(channels, requested_heads):
    heads = min(channels, requested_heads)
    while channels % heads:
        heads -= 1
    return heads

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        self.image_attention = nn.MultiheadAttention(
            channels, _valid_heads(channels, heads), batch_first=True
        )
        # A vector is treated as a sequence of scalar features.
        self.vector_attention = nn.MultiheadAttention(1, 1, batch_first=True)

    def forward(self, x):
        if x.ndim == 4:
            batch, channels, height, width = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            attended, _ = self.image_attention(tokens, tokens, tokens, need_weights=False)
            return x + attended.transpose(1, 2).reshape(batch, channels, height, width)
        tokens = x.unsqueeze(-1)
        attended, _ = self.vector_attention(tokens, tokens, tokens, need_weights=False)
        return x + attended.squeeze(-1)

` : ""}${hasCrossAttention ? `class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        self.image_attention = nn.MultiheadAttention(
            channels, _valid_heads(channels, heads), batch_first=True
        )
        self.vector_attention = nn.MultiheadAttention(1, 1, batch_first=True)

    def forward(self, query, context):
        if query.ndim == 4:
            batch, channels, height, width = query.shape
            query_tokens = query.flatten(2).transpose(1, 2)
            context_tokens = context.flatten(2).transpose(1, 2)
            attended, _ = self.image_attention(
                query_tokens, context_tokens, context_tokens, need_weights=False
            )
            return query + attended.transpose(1, 2).reshape(batch, channels, height, width)
        query_tokens, context_tokens = query.unsqueeze(-1), context.unsqueeze(-1)
        attended, _ = self.vector_attention(
            query_tokens, context_tokens, context_tokens, need_weights=False
        )
        return query + attended.squeeze(-1)

` : ""}# ${dataset.name}: ${dataset.channels} × ${dataset.size} × ${dataset.size}, ${dataset.classes} classes

# ${dataset.name}: ${dataset.channels} × ${dataset.size} × ${dataset.size}, ${dataset.classes} classes
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.${
  dataset.name === "CIFAR-10"
    ? "CIFAR10"
    : dataset.name.replace("-", "")
}(
    root="./data", train=True, download=True, transform=transform
)

train_loader = DataLoader(
    train_set,
    batch_size=${training.batchSize},
    shuffle=True
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ${definitions.join("\n        ") || "pass"}

    def forward(self, x):
        features = {}  # diagram layer outputs: features[1], features[2], ...
        ${forward.join("\n        ")}
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Net().to(device)

optimizer = ${optimizerCall}

criterion = nn.CrossEntropyLoss()

for epoch in range(${training.epochs}):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    print(
        f"epoch {epoch + 1:02d} | "
        f"loss {running_loss / total:.4f} | "
        f"accuracy {correct / total:.2%}"
    )`;
}

export default App;
