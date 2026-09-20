# NeuralGraph — Visual Neural Network Design Studio

NeuralGraph is a single-page React application that lets you **visually build a
convolutional neural network**, inspect the tensor shape at every stage, catch
architecture mistakes before you train anything, and export a ready-to-run
**PyTorch** training script.

Think of it as a drag-and-drop CNN "IDE": you assemble layers on a canvas, the
app infers the output shape of each layer in real time, flags shape mismatches
(e.g. mismatched residual branches, a `Dense` layer fed an image tensor,
`GroupNorm` with an invalid group count), and continuously regenerates working
PyTorch code you can copy or hand-edit.

---

## Features

- **Drag-and-drop architecture canvas** — add layers by clicking a toolbox
  button (inserts after the selected layer) or by dragging a tool/layer card
  directly onto another layer to place it before/after.
- **Live shape inference** — every layer shows its input → output tensor shape
  (`C × H × W` for feature maps, `N` for vectors), computed by walking the
  layer list with the correct formulas for convolution, pooling, transposed
  convolution, flattening, etc.
- **Design checks** — a rules engine flags issues such as:
  - kernels/pooling windows larger than the current feature map
  - dense layers applied before `Flatten`
  - normalization/regularization layers used on the wrong tensor rank
  - branch operations (`Add`, `Concatenate`, attention, …) with missing or
    incompatible connections
  - a final output size that doesn't match the selected dataset's class count
- **Branch / graph connections** — layers like `Add`, `Multiply`,
  `Concatenate`, `Residual`, `Skip`, `Split`, `Merge`, and attention blocks can
  be wired to any earlier layer to build residual or multi-branch topologies,
  not just a linear stack.
- **50+ layer types** across seven categories: input & data, feature
  extraction, activations, normalization, regularization, classifier heads,
  connections, and attention/advanced blocks (see full list below).
- **Activation microscope** — an interactive curve plot for any activation
  layer, with a draggable input slider that shows the live output value.
- **Dataset presets** — MNIST, Fashion-MNIST, and CIFAR-10, each setting the
  correct input shape and class count for shape inference.
- **Training configuration panel** — optimizer (AdamW / SGD / RMSprop),
  learning rate, batch size, and epoch count, all reflected in the generated
  script.
- **Live PyTorch code generation** — the canvas state compiles to a complete,
  runnable `nn.Module` + training loop, editable in place (edits are kept
  separate from the auto-generated version until you reset).

---

## Tech Stack

- **React** (functional components, hooks: `useState`, `useMemo`, `useRef`)
- Plain CSS (`App.css`, not included in this file) for styling
- No external state-management or drag-and-drop libraries — layer reordering
  and tool placement use the native HTML5 Drag and Drop API
  (`draggable`, `onDragStart`, `onDrop`, `dataTransfer`)
- Generated output targets **PyTorch** (`torch`, `torch.nn`, `torchvision`)

---

## Getting Started

```bash
# inside your React app (e.g. Vite or Create React App project)
npm install
npm run dev      # or `npm start` depending on your tooling
```

Drop `App.jsx` (or rename to `App.js`) and its companion `App.css` into your
project's `src/` directory as the app's root component.

---

## How It Works

### 1. Layer library (`LAYER_LIBRARY`)

A single lookup table defines every available layer: its display label, the
toolbox group it belongs to, and its default parameters. Layers are grouped
automatically into `LAYER_GROUPS` for rendering the sidebar toolbox.

| Group | Example layers |
|---|---|
| Input & data | Input, Normalize, Augmentation, Reshape, Flatten |
| Feature extraction | Conv2d, Depthwise Conv, Dilated Conv, 1×1 Conv, Transposed Conv, Max/Average/Adaptive/Global Pool |
| Activations | ReLU, Leaky ReLU, GELU, SiLU, ELU, SELU, Tanh, Sigmoid, Softplus |
| Normalization | BatchNorm, LayerNorm, GroupNorm, InstanceNorm |
| Regularization | Dropout, Dropout2D, DropPath |
| Classifier | Dense/Linear, Bilinear, Classifier/Classification/Regression Head |
| Connections | Add, Multiply, Concatenate, Residual, Skip, Split, Merge |
| Attention / advanced | Self/Multi-Head/Cross Attention, Channel/Spatial Attention, SE Block, CBAM |
| Output | Softmax, LogSoftmax |

### 2. Shape inference (`inferArchitecture`)

Given the ordered layer list, the current dataset, and any branch
connections, `inferArchitecture` walks the list once, tracking a `shape`
object of the form `{ kind: "image", c, h, w }` or `{ kind: "vector", n }`.
For each layer it:

1. Computes the new shape using the standard convolution/pooling output-size
   formula, `floor((in + 2*pad - dilation*(kernel-1) - 1) / stride) + 1`, or
   the layer's specific transformation (flatten, reshape, concatenate, etc.).
2. Resolves any branch connection (for `Add`/`Concatenate`/attention/etc.) by
   looking up the referenced layer's already-computed output shape.
3. Attaches a human-readable `note` describing any incompatibility found
   (e.g. *"Branch shape 64 × 7 × 7 must match 128 × 7 × 7."*).

The function returns both the annotated layer list (with `input`, `output`,
and `note` per layer) and a flat `issues` array used to render the **Design
checks** panel.

### 3. Connection integrity on reorder

Because branch connections are directional (`from` must precede `to`), both
`moveLayer` (single-step reorder) and `dropLayer` (drag-and-drop reorder)
recompute the full list order first, then **prune any connection that would
now point backward or to a removed layer**, rather than only checking the
two swapped positions.

### 4. Code generation (`makeCode`)

`makeCode` mirrors the same layer traversal to emit:

- `nn.Module.__init__` layer definitions (e.g. `nn.Conv2d(...)`,
  `nn.LazyLinear(...)`) with channel counts threaded through automatically.
- A `forward()` method that applies each layer in order, storing every
  layer's output in a `features` dict (keyed by diagram position) so branch
  operations can reference `features[i]` directly.
- Custom `SelfAttentionBlock` / `CrossAttentionBlock` helper classes,
  auto-included only when the diagram actually uses attention layers, that
  handle both image tensors (`B, C, H, W`) and vector tensors transparently
  and auto-clamp the requested head count to a valid divisor of the channel
  count.
- A complete, dataset-matched training loop (`DataLoader`, optimizer,
  `CrossEntropyLoss`, per-epoch loss/accuracy logging).

The generated code is shown in an editable `<textarea>`; hand edits are kept
in a separate `editableCode` buffer (`codeEdited` flag) so "Reset generated"
can always restore the auto-generated version without losing your place.

### 5. Activation microscope

For any selected activation layer, `ActivationLab` renders an SVG plot of the
activation function over `[-3, 3]` using the same `activationValue` function
used to compute the curve, plus a slider that moves a live marker point along
the curve — useful for building intuition about ReLU vs. LeakyReLU vs. GELU,
etc.

---

## Design Checks Reference

| Situation | Message |
|---|---|
| Conv/Pool applied to a non-image tensor | *"Convolutions need an image-shaped input."* / *"Pooling needs an image-shaped input."* |
| Kernel/pool window larger than the feature map | *"Kernel is larger than the current feature map."* |
| Dense-family layer before `Flatten` | *"Add Flatten before the first dense layer."* |
| `Flatten` applied twice | *"This is already a vector; Flatten is not needed."* |
| `GroupNorm` groups don't divide channels | *"Channel count (C) must be divisible by groups (G)."* |
| Norm/regularization layer on the wrong rank | Suggests the vector- or image-specific alternative |
| Branch layer with no connection | *"Choose an earlier layer, then attach its branch here."* |
| Incompatible branch shapes | *"Branch shape … must match …."* |
| Final layer size ≠ dataset class count | *"Your last layer has N outputs, but \<dataset\> needs K class scores."* |
| Diagram uses `Softmax` | Tip that `CrossEntropyLoss` already applies softmax internally |

---

## Known Limitations

- Shape inference assumes square inputs/kernels (`h === w`, single `kernel`
  value) — non-square configurations aren't modeled.
- `ChannelAttention`, `SpatialAttention`, `SEBlock`, and `CBAM` are validated
  for shape compatibility but emit a placeholder comment in the generated
  code rather than a full implementation.
- `LayerNorm` uses a generic `F.layer_norm(x, x.shape[1:])` rather than a
  learnable, declared module.
- `DropPath` is approximated in code as `nn.Dropout` (true stochastic depth
  is not implemented).
- No persistence layer — refreshing the page resets the canvas to
  `INITIAL_LAYERS`.

---

## Possible Extensions

- Save/load architectures (e.g. to `localStorage` or a backend) as JSON.
- Export to additional frameworks (Keras/TensorFlow, ONNX).
- Parameter-count and FLOPs estimation per layer and total.
- Undo/redo history for canvas edits.
- Validation for non-square spatial dimensions.
