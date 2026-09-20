import { useMemo, useState } from "react";

const DATASETS = {
  MNIST: {
    name: "MNIST",
    channels: 1,
    size: 28,
    classes: 10,
    normalize: "((0.1307,), (0.3081,))",
  },
  "Fashion-MNIST": {
    name: "Fashion-MNIST",
    channels: 1,
    size: 28,
    classes: 10,
    normalize: "((0.2860,), (0.3530,))",
  },
  "CIFAR-10": {
    name: "CIFAR-10",
    channels: 3,
    size: 32,
    classes: 10,
    normalize: "((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))",
  },
};

const INITIAL_LAYERS = [
  {
    id: 1,
    type: "Conv2d",
    params: {
      out_channels: 32,
      kernel_size: 3,
      padding: 1,
    },
  },
  {
    id: 2,
    type: "ReLU",
    params: {},
  },
  {
    id: 3,
    type: "MaxPool2d",
    params: {
      kernel_size: 2,
    },
  },
  {
    id: 4,
    type: "Flatten",
    params: {},
  },
  {
    id: 5,
    type: "Linear",
    params: {
      out_features: 128,
    },
  },
  {
    id: 6,
    type: "ReLU",
    params: {},
  },
  {
    id: 7,
    type: "Linear",
    params: {
      out_features: 10,
    },
  },
];

function App() {
  const [dataset, setDataset] = useState("MNIST");

  const [training, setTraining] = useState({
    batchSize: 128,
    epochs: 20,
    lr: 0.001,
    weightDecay: 0.0001,
    optimizer: "AdamW",
    scheduler: "CosineAnnealingLR",
  });

  const [augmentation, setAugmentation] = useState({
    randomCrop: false,
    horizontalFlip: false,
    rotation: false,
    colorJitter: false,
  });

  const [layers, setLayers] = useState(INITIAL_LAYERS);
  const [selectedId, setSelectedId] = useState(1);

  const selectedLayer = layers.find((l) => l.id === selectedId);

  const datasetInfo = DATASETS[dataset];

  function updateTraining(key, value) {
    setTraining((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function updateLayerParam(key, value) {
    setLayers((prev) =>
      prev.map((layer) =>
        layer.id === selectedId
          ? {
              ...layer,
              params: {
                ...layer.params,
                [key]: value,
              },
            }
          : layer
      )
    );
  }

  function addLayer(type) {
    const id = Date.now();

    const defaults = {
      Conv2d: {
        out_channels: 32,
        kernel_size: 3,
        padding: 1,
      },

      Linear: {
        out_features: 128,
      },

      ReLU: {},

      BatchNorm2d: {},

      MaxPool2d: {
        kernel_size: 2,
      },

      Dropout: {
        p: 0.5,
      },

      Flatten: {},
    };

    const newLayer = {
      id,
      type,
      params: defaults[type] || {},
    };

    setLayers((prev) => [...prev, newLayer]);
    setSelectedId(id);
  }

  function removeLayer() {
    if (!selectedLayer) return;

    setLayers((prev) =>
      prev.filter((layer) => layer.id !== selectedLayer.id)
    );

    setSelectedId(layers[0]?.id);
  }

  function generateTransforms() {
    const lines = [];

    if (augmentation.randomCrop) {
      if (dataset === "CIFAR-10") {
        lines.push("transforms.RandomCrop(32, padding=4)");
      } else {
        lines.push("transforms.RandomCrop(28, padding=4)");
      }
    }

    if (augmentation.horizontalFlip) {
      lines.push("transforms.RandomHorizontalFlip()");
    }

    if (augmentation.rotation) {
      lines.push("transforms.RandomRotation(10)");
    }

    if (augmentation.colorJitter && dataset === "CIFAR-10") {
      lines.push("transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)");
    }

    lines.push("transforms.ToTensor()");
    lines.push(
      `transforms.Normalize(${datasetInfo.normalize})`
    );

    return lines.join(",\n        ");
  }

  function generateDatasetCode() {
    let datasetClass = "";

    if (dataset === "MNIST") {
      datasetClass = "datasets.MNIST";
    } else if (dataset === "Fashion-MNIST") {
      datasetClass = "datasets.FashionMNIST";
    } else {
      datasetClass = "datasets.CIFAR10";
    }

    return `train_transform = transforms.Compose([
        ${generateTransforms()}
])

train_dataset = ${datasetClass}(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=${training.batchSize},
    shuffle=True,
    num_workers=4
)`;
  }

  function generateModelCode() {
    let code = `class Net(nn.Module):
    def __init__(self):
        super().__init__()

`;

    let previousChannels = datasetInfo.channels;
    let currentSpatial = datasetInfo.size;
    let flattenedFeatures = null;
    let linearCount = 0;
    let convCount = 0;
    let poolCount = 0;
    let bnCount = 0;
    let dropoutCount = 0;

    layers.forEach((layer) => {
      const p = layer.params;

      if (layer.type === "Conv2d") {
        convCount++;

        code += `        self.conv${convCount} = nn.Conv2d(
            ${previousChannels},
            ${p.out_channels},
            kernel_size=${p.kernel_size},
            padding=${p.padding}
        )\n`;

        previousChannels = Number(p.out_channels);
      }

      if (layer.type === "BatchNorm2d") {
        bnCount++;

        code += `        self.bn${bnCount} = nn.BatchNorm2d(${previousChannels})\n`;
      }

      if (layer.type === "ReLU") {
        code += `        self.relu${layer.id} = nn.ReLU()\n`;
      }

      if (layer.type === "MaxPool2d") {
        poolCount++;

        code += `        self.pool${poolCount} = nn.MaxPool2d(${p.kernel_size})\n`;

        currentSpatial = Math.floor(
          currentSpatial / Number(p.kernel_size)
        );
      }

      if (layer.type === "Dropout") {
        dropoutCount++;

        code += `        self.dropout${dropoutCount} = nn.Dropout(${p.p})\n`;
      }

      if (layer.type === "Flatten") {
        flattenedFeatures =
          previousChannels *
          currentSpatial *
          currentSpatial;

        code += `        self.flatten = nn.Flatten()\n`;
      }

      if (layer.type === "Linear") {
        linearCount++;

        let inputFeatures;

        if (linearCount === 1) {
          inputFeatures =
            flattenedFeatures ||
            previousChannels *
              currentSpatial *
              currentSpatial;
        } else {
          const previousLinear = layers
            .slice(0, layers.indexOf(layer))
            .reverse()
            .find((x) => x.type === "Linear");

          inputFeatures =
            previousLinear?.params?.out_features || 128;
        }

        code += `        self.fc${linearCount} = nn.Linear(
            ${inputFeatures},
            ${p.out_features}
        )\n`;
      }
    });

    code += `
    def forward(self, x):
`;

    let linearIndex = 0;
    let convIndex = 0;
    let poolIndex = 0;
    let bnIndex = 0;
    let dropoutIndex = 0;

    layers.forEach((layer) => {
      const p = layer.params;

      if (layer.type === "Conv2d") {
        convIndex++;
        code += `        x = self.conv${convIndex}(x)\n`;
      }

      if (layer.type === "BatchNorm2d") {
        bnIndex++;
        code += `        x = self.bn${bnIndex}(x)\n`;
      }

      if (layer.type === "ReLU") {
        code += `        x = self.relu${layer.id}(x)\n`;
      }

      if (layer.type === "MaxPool2d") {
        poolIndex++;
        code += `        x = self.pool${poolIndex}(x)\n`;
      }

      if (layer.type === "Flatten") {
        code += `        x = self.flatten(x)\n`;
      }

      if (layer.type === "Dropout") {
        dropoutIndex++;
        code += `        x = self.dropout${dropoutIndex}(x)\n`;
      }

      if (layer.type === "Linear") {
        linearIndex++;
        code += `        x = self.fc${linearIndex}(x)\n`;
      }
    });

    code += `
        return x
`;

    return code;
  }

  function generateOptimizerCode() {
    const optimizerMap = {
      SGD: `torch.optim.SGD(
    model.parameters(),
    lr=${training.lr},
    weight_decay=${training.weightDecay},
    momentum=0.9
)`,

      Adam: `torch.optim.Adam(
    model.parameters(),
    lr=${training.lr},
    weight_decay=${training.weightDecay}
)`,

      AdamW: `torch.optim.AdamW(
    model.parameters(),
    lr=${training.lr},
    weight_decay=${training.weightDecay}
)`,
    };

    return `optimizer = ${optimizerMap[training.optimizer]}

scheduler = torch.optim.lr_scheduler.${training.scheduler}(
    optimizer,
    T_max=${training.epochs}
)`;
  }

  const fullCode = useMemo(() => {
    return `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

${generateDatasetCode()}


${generateModelCode()}


model = Net()

${generateOptimizerCode()}

EPOCHS = ${training.epochs}

`;
  }, [
    dataset,
    training,
    augmentation,
    layers,
  ]);

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* HEADER */}

      <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">
            NeuralGraph
          </h1>

          <p className="text-sm text-slate-400">
            Interactive neural network experimenter
          </p>
        </div>

        <button className="bg-blue-600 hover:bg-blue-500 px-5 py-2 rounded-lg font-medium">
          ▶ Train Model
        </button>
      </header>

      <main className="grid grid-cols-[260px_1fr_320px] min-h-[calc(100vh-80px)]">

        {/* LEFT PANEL */}

        <aside className="border-r border-slate-800 p-5 space-y-7 overflow-y-auto">

          <section>
            <h2 className="font-semibold mb-3">
              Dataset
            </h2>

            <select
              value={dataset}
              onChange={(e) =>
                setDataset(e.target.value)
              }
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2"
            >
              <option>MNIST</option>
              <option>Fashion-MNIST</option>
              <option>CIFAR-10</option>
            </select>

            <div className="mt-3 text-sm text-slate-400 space-y-1">
              <div>
                Input:{" "}
                <span className="text-white">
                  {datasetInfo.channels} ×{" "}
                  {datasetInfo.size} ×{" "}
                  {datasetInfo.size}
                </span>
              </div>

              <div>
                Classes:{" "}
                <span className="text-white">
                  {datasetInfo.classes}
                </span>
              </div>
            </div>
          </section>

          <section>
            <h2 className="font-semibold mb-3">
              Add Layer
            </h2>

            <div className="grid grid-cols-2 gap-2">
              {[
                "Conv2d",
                "Linear",
                "ReLU",
                "BatchNorm2d",
                "MaxPool2d",
                "Dropout",
                "Flatten",
              ].map((type) => (
                <button
                  key={type}
                  onClick={() => addLayer(type)}
                  className="bg-slate-900 hover:bg-slate-800 border border-slate-700 rounded-lg px-2 py-2 text-sm"
                >
                  + {type}
                </button>
              ))}
            </div>
          </section>

          <section>
            <h2 className="font-semibold mb-3">
              Training Augmentation
            </h2>

            {Object.entries(augmentation).map(
              ([key, value]) => (
                <label
                  key={key}
                  className="flex items-center gap-3 py-2 text-sm"
                >
                  <input
                    type="checkbox"
                    checked={value}
                    onChange={(e) =>
                      setAugmentation((prev) => ({
                        ...prev,
                        [key]: e.target.checked,
                      }))
                    }
                  />

                  {key
                    .replace(/([A-Z])/g, " $1")
                    .replace(/^./, (x) =>
                      x.toUpperCase()
                    )}
                </label>
              )
            )}
          </section>

        </aside>

        {/* CENTER */}

        <section className="flex flex-col">

          <div className="border-b border-slate-800 px-6 py-4 flex justify-between">
            <div>
              <h2 className="font-semibold">
                Computation Graph
              </h2>

              <p className="text-sm text-slate-400">
                Click a layer to modify it
              </p>
            </div>

            <div className="text-sm text-slate-400">
              {layers.length} nodes
            </div>
          </div>

          <div className="flex-1 p-8 overflow-auto">

            <div className="max-w-md mx-auto space-y-3">

              <div className="border border-blue-500 bg-blue-500/10 rounded-xl p-4 text-center">
                Input

                <div className="text-sm text-slate-400 mt-1">
                  [{datasetInfo.channels},{" "}
                  {datasetInfo.size},{" "}
                  {datasetInfo.size}]
                </div>
              </div>

              {layers.map((layer) => (
                <div key={layer.id}>

                  <div className="flex justify-center text-slate-500">
                    ↓
                  </div>

                  <button
                    onClick={() =>
                      setSelectedId(layer.id)
                    }
                    className={`w-full text-left border rounded-xl p-4 transition ${
                      selectedId === layer.id
                        ? "border-blue-500 bg-blue-500/10"
                        : "border-slate-700 bg-slate-900 hover:border-slate-500"
                    }`}
                  >
                    <div className="font-medium">
                      {layer.type}
                    </div>

                    <div className="text-sm text-slate-400 mt-1">
                      {layer.type === "Conv2d" &&
                        `out_channels=${layer.params.out_channels}, kernel=${layer.params.kernel_size}`}

                      {layer.type === "Linear" &&
                        `out_features=${layer.params.out_features}`}

                      {layer.type === "MaxPool2d" &&
                        `kernel=${layer.params.kernel_size}`}

                      {layer.type === "Dropout" &&
                        `p=${layer.params.p}`}
                    </div>
                  </button>

                </div>
              ))}

              <div className="flex justify-center text-slate-500">
                ↓
              </div>

              <div className="border border-green-500 bg-green-500/10 rounded-xl p-4 text-center">
                Output

                <div className="text-sm text-slate-400 mt-1">
                  {datasetInfo.classes} classes
                </div>
              </div>

            </div>

          </div>

          {/* CODE */}

          <div className="border-t border-slate-800 h-[330px]">

            <div className="px-5 py-3 border-b border-slate-800 flex justify-between">
              <span className="font-medium">
                Generated PyTorch
              </span>

              <button
                onClick={() =>
                  navigator.clipboard?.writeText(fullCode)
                }
                className="text-sm text-blue-400"
              >
                Copy
              </button>
            </div>

            <pre className="p-5 text-xs text-slate-300 overflow-auto h-full">
              {fullCode}
            </pre>

          </div>

        </section>

        {/* RIGHT PANEL */}

        <aside className="border-l border-slate-800 p-5 overflow-y-auto">

          <h2 className="font-semibold mb-5">
            Inspector
          </h2>

          {selectedLayer && (
            <div className="space-y-5">

              <div>
                <div className="text-sm text-slate-400">
                  Layer
                </div>

                <div className="text-lg font-medium">
                  {selectedLayer.type}
                </div>
              </div>

              {selectedLayer.type === "Conv2d" && (
                <>
                  <NumberInput
                    label="Output Channels"
                    value={
                      selectedLayer.params.out_channels
                    }
                    onChange={(v) =>
                      updateLayerParam(
                        "out_channels",
                        Number(v)
                      )
                    }
                  />

                  <NumberInput
                    label="Kernel Size"
                    value={
                      selectedLayer.params.kernel_size
                    }
                    onChange={(v) =>
                      updateLayerParam(
                        "kernel_size",
                        Number(v)
                      )
                    }
                  />

                  <NumberInput
                    label="Padding"
                    value={
                      selectedLayer.params.padding
                    }
                    onChange={(v) =>
                      updateLayerParam(
                        "padding",
                        Number(v)
                      )
                    }
                  />
                </>
              )}

              {selectedLayer.type === "Linear" && (
                <NumberInput
                  label="Output Features"
                  value={
                    selectedLayer.params.out_features
                  }
                  onChange={(v) =>
                    updateLayerParam(
                      "out_features",
                      Number(v)
                    )
                  }
                />
              )}

              {selectedLayer.type === "MaxPool2d" && (
                <NumberInput
                  label="Kernel Size"
                  value={
                    selectedLayer.params.kernel_size
                  }
                  onChange={(v) =>
                    updateLayerParam(
                      "kernel_size",
                      Number(v)
                    )
                  }
                />
              )}

              {selectedLayer.type === "Dropout" && (
                <NumberInput
                  label="Probability"
                  value={
                    selectedLayer.params.p
                  }
                  step="0.1"
                  onChange={(v) =>
                    updateLayerParam(
                      "p",
                      Number(v)
                    )
                  }
                />
              )}

              <button
                onClick={removeLayer}
                className="w-full border border-red-500/50 text-red-400 hover:bg-red-500/10 rounded-lg py-2"
              >
                Delete Layer
              </button>

            </div>
          )}

          <hr className="border-slate-800 my-7" />

          <h2 className="font-semibold mb-4">
            Training
          </h2>

          <div className="space-y-4">

            <NumberInput
              label="Batch Size"
              value={training.batchSize}
              onChange={(v) =>
                updateTraining(
                  "batchSize",
                  Number(v)
                )
              }
            />

            <NumberInput
              label="Epochs"
              value={training.epochs}
              onChange={(v) =>
                updateTraining(
                  "epochs",
                  Number(v)
                )
              }
            />

            <NumberInput
              label="Learning Rate"
              value={training.lr}
              step="0.0001"
              onChange={(v) =>
                updateTraining(
                  "lr",
                  Number(v)
                )
              }
            />

            <NumberInput
              label="Weight Decay"
              value={training.weightDecay}
              step="0.0001"
              onChange={(v) =>
                updateTraining(
                  "weightDecay",
                  Number(v)
                )
              }
            />

            <SelectInput
              label="Optimizer"
              value={training.optimizer}
              options={[
                "SGD",
                "Adam",
                "AdamW",
              ]}
              onChange={(v) =>
                updateTraining(
                  "optimizer",
                  v
                )
              }
            />

            <SelectInput
              label="Scheduler"
              value={training.scheduler}
              options={[
                "CosineAnnealingLR",
                "StepLR",
                "OneCycleLR",
              ]}
              onChange={(v) =>
                updateTraining(
                  "scheduler",
                  v
                )
              }
            />

          </div>

        </aside>

      </main>
    </div>
  );
}

function NumberInput({
  label,
  value,
  onChange,
  step = "1",
}) {
  return (
    <label className="block">
      <div className="text-sm text-slate-400 mb-1">
        {label}
      </div>

      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) =>
          onChange(e.target.value)
        }
        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 outline-none focus:border-blue-500"
      />
    </label>
  );
}

function SelectInput({
  label,
  value,
  options,
  onChange,
}) {
  return (
    <label className="block">
      <div className="text-sm text-slate-400 mb-1">
        {label}
      </div>

      <select
        value={value}
        onChange={(e) =>
          onChange(e.target.value)
        }
        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2"
      >
        {options.map((option) => (
          <option key={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export default App;
