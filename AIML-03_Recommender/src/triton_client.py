"""
NVIDIA Triton Inference Server Client Bridge.
Step 7 of AIML-03: Serve models through Triton.
Provides gRPC / HTTP client connectivity with seamless local model fallback.
"""

import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8000")


class TritonInferenceBridge:
    """
    Production client for NVIDIA Triton Inference Server.
    Automatically detects server availability and falls back to local inference if Triton is offline.
    """
    def __init__(self, url: str = TRITON_SERVER_URL):
        self.url = url
        self.is_connected = False
        self.http_client = None
        self._init_client()

    def _init_client(self):
        try:
            import tritonclient.http as httpclient
            client = httpclient.InferenceServerClient(url=self.url, connection_timeout=0.5, network_timeout=0.5)
            if client.is_server_live():
                self.http_client = client
                self.is_connected = True
                print(f"[Triton] Connected to NVIDIA Triton Server at {self.url}")
        except Exception:
            self.is_connected = False
            # Operates seamlessly in local fallback mode

    def infer_user_embedding(self, user_id: str, local_embedding: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Retrieves user embedding from Triton or local cache.
        Returns: (embedding_array, source_string)
        """
        if self.is_connected and self.http_client is not None:
            try:
                import tritonclient.http as httpclient
                uid_int = int(user_id) if user_id.isdigit() else 1
                inp = httpclient.InferInput("user_id", [1, 1], "INT64")
                inp.set_data_from_numpy(np.array([[uid_int]], dtype=np.int64))
                out = httpclient.InferRequestedOutput("embedding")
                resp = self.http_client.infer(model_name="two_tower", inputs=[inp], outputs=[out])
                return resp.as_numpy("embedding")[0], "Triton-Inference-Server"
            except Exception:
                pass
        return local_embedding, "Local-Model-Runtime"

    def infer_ranking(self, feature_matrix: np.ndarray, local_predictions: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Executes candidate re-ranking on Triton or local DCN engine.
        """
        if self.is_connected and self.http_client is not None:
            try:
                import tritonclient.http as httpclient
                inp = httpclient.InferInput("features", feature_matrix.shape, "FP32")
                inp.set_data_from_numpy(feature_matrix.astype(np.float32))
                out = httpclient.InferRequestedOutput("ranking_score")
                resp = self.http_client.infer(model_name="ranking", inputs=[inp], outputs=[out])
                return resp.as_numpy("ranking_score").flatten(), "Triton-Inference-Server"
            except Exception:
                pass
        return local_predictions, "Local-DCN-Runtime"

    def get_status(self) -> Dict[str, Any]:
        return {
            "triton_url": self.url,
            "connected": self.is_connected,
            "mode": "NVIDIA Triton Production Server" if self.is_connected else "Local Optimized Runtime (Standby for Triton container)"
        }


TRITON_BRIDGE = TritonInferenceBridge()
