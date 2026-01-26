from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass

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
    def __init__(self, config: dict):
        self.config = config
        self.meta = config.get('model_meta', {})
        self.layers_data = config.get('layers', [])
        self.spec_raw = config.get('verification_spec', {}) 
        
        self.model = self.build_model()
        self.model.eval()

    def get_spec(self):
        """Dispatches parsing based on ptype (lp, robustness, etc.)"""
        problem_type = self.meta.get('ptype', 'lp').lower()
        
        parsers = {
            'lp': LPSpecParser(),
            # 'robustness': RobustnessSpecParser(),
        }
        
        parser = parsers.get(problem_type)
        if not parser:
            raise ValueError(f"No parser registered for ptype: {problem_type}")
            
        return parser.parse(self.spec_raw)

    def build_model(self) -> nn.Sequential:
        """Standardized build process for any architecture."""
        modules = []
        activation_type = self.meta.get('activation', 'relu').lower()
        
        for i, layer_cfg in enumerate(self.layers_data):
            # 1. Determine type (default to 'linear' for feedforward)
            l_type = layer_cfg.get('type', self.meta.get('architecture', 'linear'))
            
            # 2. Reconstruct the weights/layer
            reconstructor = ReconstructorFactory.get(l_type)
            modules.append(reconstructor.reconstruct(layer_cfg))
            
            # 3. Add activation if not the last layer and type is weight-based
            if i < len(self.layers_data) - 1 and l_type != 'flatten':
                modules.append(self._get_activation_layer(activation_type))
        
        return nn.Sequential(*modules)

    def _get_activation_layer(self, name: str) -> nn.Module:
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }.get(name, nn.Identity())
        
    def get_layer_params(self) -> List[dict]:
        """
        Returns the raw layer data (weights, biases, activations) 
        formatted for the MILPVerifier.
        """
        # If your MILPVerifier expects the raw list of dictionaries:
        return self.layers_data






















# import torch
# import torch.nn as nn
# import yaml

# class NNLoader:
#     def __init__(self, config):        
#         self.config = config
#         self.meta = self.config.get('model_meta', {})
#         self.architecture = self.meta.get('architecture', 'feedforward').lower()
#         self.activation = self.meta.get('activation', 'relu').lower()
#         self.layers_data = self.config['layers']
#         self.spec = self.config.get('verification_spec', {})

#     def build_model(self):
#         """
#         Main entry point that dispatches to the correct builder based on arch.
#         """
#         if self.architecture == 'feedforward':
#             return self._build_feedforward()
#         elif self.architecture == 'cnn':
#             return self._build_cnn()
#         else:
#             raise ValueError(f"Unsupported architecture: {self.architecture}")

#     def _build_feedforward(self):
#         """Reconstructs a MLP / Linear model."""
#         modules = []
#         for i, layer_cfg in enumerate(self.layers_data):
#             w = torch.tensor(layer_cfg['weights'], dtype=torch.float32)
#             b = torch.tensor(layer_cfg['biases'], dtype=torch.float32)
            
#             out_f, in_f = w.shape
#             linear = nn.Linear(in_f, out_f)
            
#             with torch.no_grad():
#                 linear.weight.copy_(w)
#                 linear.bias.copy_(b)
            
#             modules.append(linear)
#             if i < len(self.layers_data) - 1:
#                 modules.append(self._get_activation())
        
#         return nn.Sequential(*modules)

#     def _build_cnn(self):
#         """
#         Skeleton for CNN reconstruction. 
#         CNNs require 'kernel_size', 'stride', and 'padding' in the config.
#         """
#         modules = []
#         # Logic for CNNs usually involves alternating Conv2d and Linear (at the end)
#         for i, layer_cfg in enumerate(self.layers_data):
#             layer_type = layer_cfg.get('type', 'conv2d')
            
#             if layer_type == 'conv2d':
#                 # Weights for Conv2d are [Out_Channels, In_Channels, K_H, K_W]
#                 w = torch.tensor(layer_cfg['weights'], dtype=torch.float32)
#                 b = torch.tensor(layer_cfg['biases'], dtype=torch.float32)
                
#                 conv = nn.Conv2d(
#                     in_channels=w.shape[1],
#                     out_channels=w.shape[0],
#                     kernel_size=layer_cfg.get('kernel_size', 3),
#                     stride=layer_cfg.get('stride', 1),
#                     padding=layer_cfg.get('padding', 0)
#                 )
#                 with torch.no_grad():
#                     conv.weight.copy_(w)
#                     conv.bias.copy_(b)
#                 modules.append(conv)
#                 modules.append(self._get_activation())

#             elif layer_type == 'flatten':
#                 modules.append(nn.Flatten())

#             elif layer_type == 'linear':
#                 # Similar to feedforward logic
#                 pass 
                
#         return nn.Sequential(*modules)

#     def _get_activation(self):
#         """Helper to return the correct activation layer."""
#         if self.activation == 'relu':
#             return nn.ReLU()
#         elif self.activation == 'tanh':
#             return nn.Tanh()
#         return nn.Identity()

#     def get_spec(self):
#         """
#         Parses the canonical verification parameters.
#         Returns: Dict containing input_bounds (as list of pairs), A, and c.
#         """
#         raw_bounds = self.spec.get("input_bounds", [])
        
#         # Safe extraction: Handles the new 'min'/'max' dictionary format
#         # If raw_bounds is already [[l, h], ...], this will need a check
#         if raw_bounds and isinstance(raw_bounds[0], dict):
#             formatted_bounds = [[b['min'], b['max']] for b in raw_bounds]
#         else:
#             # Fallback for old format or empty list
#             formatted_bounds = raw_bounds

#         constraints = self.spec.get("constraints", {})
        
#         return {
#             "input_bounds": formatted_bounds,
#             "A": constraints.get("A", []),
#             "b_limit": constraints.get("b_limit", []),
#             "c": self.spec.get("objective_c", [])
#         }

#     def get_layer_params(self):
#         """Returns raw weight/bias dictionaries for the MILP engine."""
#         return self.layers_data