NVIDIA Triton Inference Server Configuration

This directory contains the configuration files required to deploy the Two-Tower Recommendation model using NVIDIA Triton Inference Server in a production environment.

📁 Directory Structure
text
triton_models/
└── two_tower/
    ├── 1/
    │   └── model.onnx       # Exported ONNX model (generated via tf2onnx)
    └── config.pbtxt         # Triton model configuration

🛠️ Why Triton?

NVIDIA Triton Inference Server enables high-performance, GPU-accelerated model inference in production. Key features include:

Dynamic Batching: Automatically groups incoming requests to maximize throughput.
Multi-Framework Support: Supports TensorFlow, PyTorch, ONNX Runtime, and other inference backends.
Prometheus Metrics: Provides built-in monitoring for latency and throughput.
Model Ensembles: Can combine the Two-Tower retrieval model with a ranking model into an inference pipeline.
📄 config.pbtxt Overview

The config.pbtxt file defines how Triton loads and serves the model, including:

Model name: two_tower
Backend/platform: onnxruntime_onnx
Input shapes: User IDs and item features, as defined by the exported model.
Output shape: 32-dimensional embeddings, assuming the exported model produces these embeddings.
Maximum batch size: 128, depending on the model configuration.

The input and output names, data types, and dimensions must match the exported ONNX model.

🚀 How to Export and Deploy
1. Export the Model to ONNX (Google Colab)

Install the required conversion dependencies:

python
!pip install tf2onnx onnx


Use the following example as a starting point for exporting a trained TensorFlow model:

python
import tensorflow as tf
import tf2onnx

# Load your trained TensorFlow model.
model = tf.keras.models.load_model("/path/to/saved_model")

# Define the input signature to match your model.
input_signature = (
    tf.TensorSpec((None, 1), tf.int64, name="user_id"),
)

# Export the model to ONNX.
tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path="model.onnx",
)


Copy the exported model to the Triton model repository:

text
triton_models/
└── two_tower/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt


Note: Update the model path, input signature, and conversion code to match your actual Two-Tower model. If your model has separate user and item towers, export the appropriate inference model with the expected inputs and embedding output.

2. Start Triton Server Using Docker

Run the following command from the project root directory:

bash
docker run --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$(pwd)/triton_models:/models" \
  nvcr.io/nvidia/tritonserver:23.12-py3 \
  tritonserver --model-repository=/models


The exposed ports are:

8000: HTTP inference API
8001: gRPC inference API
8002: Metrics endpoint

Ensure Docker, NVIDIA GPU support, and the appropriate NVIDIA Container Toolkit are installed if you intend to run Triton with GPU acceleration.

3. Send an Inference Request to Triton

Install the Triton HTTP client:

bash
pip install tritonclient[http]


Example Python client:

python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(
    url="localhost:8000"
)

inputs = httpclient.InferInput(
    "user_id",
    [1, 1],
    "INT64",
)

inputs.set_data_from_numpy(
    np.array([[1]], dtype=np.int64)
)

outputs = [
    httpclient.InferRequestedOutput("embedding")
]

response = client.infer(
    model_name="two_tower",
    inputs=[inputs],
    outputs=outputs,
)

embedding = response.as_numpy("embedding")
print(embedding)


Important: The input name (user_id), output name (embedding), tensor data type, and tensor shape are examples. They must match the actual input and output definitions of your exported ONNX model and config.pbtxt.

🧠 Local Capstone Demo vs. Production Deployment

For the local capstone demo, the FastAPI backend (src/api.py) loads the .npy embedding files directly into memory and uses FAISS for retrieval.

In a production cloud deployment, FastAPI can call NVIDIA Triton Inference Server to generate embeddings and serve model inference instead of loading precomputed .npy files directly.

This setup separates model inference from the API layer and provides a path toward scalable deployment, monitoring, and GPU-accelerated inference.

📌 Notes
Ensure config.pbtxt matches the exported ONNX model's input and output specifications.
Place the ONNX model inside the versioned directory two_tower/1/.
Triton expects the model repository to follow its standard directory structure.
The Docker image version shown is an example; use a Triton version compatible with your exported model and deployment environment.
The export and inference examples must be adapted to the actual architecture of your Two-Tower recommendation model.