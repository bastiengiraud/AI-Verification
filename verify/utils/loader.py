from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

class LayerReconstructor(ABC):
    """Abstract strategy for converting config data into PyTorch modules."""
    @abstractmethod
    def reconstruct(self, data: dict) -> nn.Module:
        pass

class LinearReconstructor(LayerReconstructor):
    def reconstruct(self, data: dict) -> nn.Module:
        w = torch.tensor(data['weights'], dtype=torch.float32)
        b = torch.tensor(data['biases'], dtype=torch.float32)
        layer = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            layer.weight.copy_(w)
            layer.bias.copy_(b)
        return layer

class Conv2dReconstructor(LayerReconstructor):
    def reconstruct(self, data: dict) -> nn.Module:
        w = torch.tensor(data['weights'], dtype=torch.float32)
        b = torch.tensor(data['biases'], dtype=torch.float32)
        conv = nn.Conv2d(
            in_channels=w.shape[1],
            out_channels=w.shape[0],
            kernel_size=data.get('kernel_size', 3),
            stride=data.get('stride', 1),
            padding=data.get('padding', 0)
        )
        with torch.no_grad():
            conv.weight.copy_(w)
            conv.bias.copy_(b)
        return conv

class FlattenReconstructor(LayerReconstructor):
    def reconstruct(self, data: dict) -> nn.Module:
        return nn.Flatten()


class ReconstructorFactory:
    _registry = {
        'linear': LinearReconstructor,
        'conv2d': Conv2dReconstructor,
        'flatten': FlattenReconstructor,
        'feedforward': LinearReconstructor # Alias
    }

    @classmethod
    def get(cls, type_name: str) -> LayerReconstructor:
        reconstructor_class = cls._registry.get(type_name.lower())
        if not reconstructor_class:
            raise ValueError(f"Unknown layer type: {type_name}")
        return reconstructor_class()

class VerificationSpec:
    def __init__(self, input_bounds, constraints_A, objective_c, b_static, input_indices, output_indices):
        self.input_bounds = input_bounds  # List of (L, U) tuples
        self.constraints_A = constraints_A
        self.objective_c = objective_c
        self.b_static = b_static
        self.input_indices = input_indices
        self.output_indices = output_indices

    @property
    def input_center(self):
        """Calculates the center of the input box."""
        return [(b[1] + b[0]) / 2.0 for b in self.input_bounds]

    @property
    def input_radius(self):
        """Calculates the radius (epsilon) of the input box."""
        return [(b[1] - b[0]) / 2.0 for b in self.input_bounds]
               
               
class SpecParser(ABC):
    @abstractmethod
    def parse(self, spec_data: dict) -> Any:
        pass

    def _parse_bounds(self, raw_bounds):
        parsed = []
        for b in raw_bounds:
            try:
                # Look for dictionary keys 'min' and 'max'
                lower = float(b.get('min', b.get('lower', 0.0)))
                upper = float(b.get('max', b.get('upper', 1.0)))
                parsed.append((lower, upper))
            except (AttributeError, TypeError):
                # Fallback if the YAML structure is actually [0, 5] instead of min/max
                try:
                    parsed.append((float(b[0]), float(b[1])))
                except:
                    print(f"[-] ERROR: Could not parse bound entry: {b}")
                    raise
        return parsed

class LPSpecParser(SpecParser):
    def parse(self, spec_data: dict) -> VerificationSpec:
        # 1. Parse Bounds: Updated to handle dictionary format {min: X, max: Y}
        raw_bounds = spec_data.get("input_bounds", [])
        bounds = []
        for b in raw_bounds:
            # Handle dictionary format (min/max) or fallback to list format [0, 5]
            if isinstance(b, dict):
                low = b.get('min', b.get('lower', 0.0))
                high = b.get('max', b.get('upper', 1.0))
            else:
                low, high = b[0], b[1]
            bounds.append((float(low), float(high)))
        
        # 2. Extract Mapping Indices
        mapping = spec_data.get("indices", {})
        in_idx = mapping.get("input_indices", [])
        out_idx = mapping.get("output_indices", [])
        
        # 3. Extract Physics
        constraints = spec_data.get("constraints", {})
        A = np.array(constraints.get("A", []), dtype=np.float32)
        b_static = np.array(constraints.get("b_static", []), dtype=np.float32)
        c = np.array(spec_data.get("objective_c", []), dtype=np.float32)
        
        return VerificationSpec(
            input_bounds=bounds,
            constraints_A=A,
            objective_c=c,
            b_static=b_static,
            input_indices=in_idx,
            output_indices=out_idx
        )

# Example of how easy it is to add a new type later
class RobustnessSpecParser(SpecParser):
    def parse(self, spec_data: dict):
        # Logic for epsilon-ball or adversarial perturbations
        pass               

class NNLoader:
    def __init__(self, config: dict, root_dir: Path = None):
        self.config = config
        self.meta = config.get('model_meta', {})
        self.proxy_spec = config.get('proxy_spec', {})
        self.spec_raw = config.get('verification_spec', {})
        
        # Determine the root directory to find the /models folder
        if root_dir is None:
            # We are in ROOT/verify/nn_loader.py, so we need to go up ONE level
            self.root_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.root_dir = Path(root_dir).resolve()
            
        # print(f"[*] Root directory identified as: {self.root_dir}")
        
        # New: Load from .pt file
        self.model = self.load_from_file()
        self.model.eval()

    def load_from_file(self) -> nn.Module:
        rel_path = self.proxy_spec.get('nn_path')
        full_path = self.root_dir / rel_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Model file not found at: {full_path}")
        
        # We set weights_only=False because we saved the full model object, 
        # and we trust our own training script.
        model = torch.load(full_path, map_location=torch.device('cpu'), weights_only=False)
        
        return model

    def get_layer_params(self) -> List[dict]:
        layers_data = []
        
        # .children() only looks at the immediate sub-layers of the model
        # This prevents 'named_modules()' from returning the model itself as a layer
        for module in self.model.children():
            if isinstance(module, nn.Linear):
                layers_data.append({
                    "weights": module.weight.detach().numpy(),
                    "biases": module.bias.detach().numpy(),
                    "type": "linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features
                })
        return layers_data

    def get_spec(self):
        """Dispatches parsing based on ptype (Unchanged from your original)"""
        problem_type = self.meta.get('ptype', 'lp').lower()
        parsers = {'lp': LPSpecParser()}
        
        parser = parsers.get(problem_type)
        if not parser:
            raise ValueError(f"No parser registered for ptype: {problem_type}")
            
        return parser.parse(self.spec_raw)













