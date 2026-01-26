from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import pyomo.environ as aml
from pyomo.opt import SolverFactory, TerminationCondition
import numpy as np


# =============================================================================
# 1. CONFIGURATION CLASSES (Separate concerns)
# =============================================================================

@dataclass
class VerifierConfig:
    """Central configuration for verifier behavior."""
    big_m: float = 1e6
    bound_u: float = 1000.0
    solver_name: str = 'glpk'
    solver_options: Dict[str, Any] = field(default_factory=dict)
    use_obbt: bool = True
    verbose: bool = False
    numerical_tolerance: float = 1e-6


@dataclass
class LayerInfo:
    """Structured layer information."""
    weights: np.ndarray
    biases: np.ndarray
    activation: str = 'relu'
    
    @property
    def num_neurons(self) -> int:
        return len(self.biases)
    
    @property
    def input_dim(self) -> int:
        return self.weights.shape[1]


# =============================================================================
# 2. ACTIVATION FUNCTION STRATEGY (Easy to extend)
# =============================================================================

class ActivationEncoder(ABC):
    """Abstract base for activation function encoding."""
    
    @abstractmethod
    def encode(self, model, layer_idx: int, neuron_idx: int, 
               lin_expr, bounds: Tuple[float, float], 
               config: VerifierConfig) -> Dict[str, Any]:
        """
        Encode activation constraints into the model.
        
        Returns:
            Dict with keys: 'fixed_on', 'fixed_off', 'unstable'
        """
        pass


class ReLUEncoder(ActivationEncoder):
    """ReLU activation with Big-M encoding."""
    
    def encode(self, model, layer_idx: int, neuron_idx: int,
               lin_expr, bounds: Tuple[float, float],
               config: VerifierConfig) -> Dict[str, Any]:
        L_i, U_i = bounds
        
        if L_i >= -config.numerical_tolerance:
            # Always ON
            model.nn_cons.add(model.y[layer_idx, neuron_idx] == lin_expr)
            model.z[layer_idx, neuron_idx].fix(1)
            model.y[layer_idx, neuron_idx].setlb(max(0, L_i))
            model.y[layer_idx, neuron_idx].setub(U_i)
            return {'fixed_on': 1, 'fixed_off': 0, 'unstable': 0}
        
        elif U_i <= config.numerical_tolerance:
            # Always OFF
            model.nn_cons.add(model.y[layer_idx, neuron_idx] == 0)
            model.z[layer_idx, neuron_idx].fix(0)
            model.y[layer_idx, neuron_idx].setlb(0)
            model.y[layer_idx, neuron_idx].setub(0)
            return {'fixed_on': 0, 'fixed_off': 1, 'unstable': 0}
        
        else:
            # Unstable - Big-M constraints
            model.nn_cons.add(model.y[layer_idx, neuron_idx] >= lin_expr)
            model.nn_cons.add(model.y[layer_idx, neuron_idx] >= 0)
            model.nn_cons.add(
                model.y[layer_idx, neuron_idx] <= 
                lin_expr + abs(L_i) * (1 - model.z[layer_idx, neuron_idx])
            )
            model.nn_cons.add(
                model.y[layer_idx, neuron_idx] <= 
                U_i * model.z[layer_idx, neuron_idx]
            )
            model.y[layer_idx, neuron_idx].setlb(0)
            model.y[layer_idx, neuron_idx].setub(U_i)
            return {'fixed_on': 0, 'fixed_off': 0, 'unstable': 1}


class LinearEncoder(ActivationEncoder):
    """Identity/Linear activation (for output layer)."""
    
    def encode(self, model, layer_idx: int, neuron_idx: int,
               lin_expr, bounds: Tuple[float, float],
               config: VerifierConfig) -> Dict[str, Any]:
        L_i, U_i = bounds
        model.nn_cons.add(model.y[layer_idx, neuron_idx] == lin_expr)
        model.y[layer_idx, neuron_idx].setlb(L_i)
        model.y[layer_idx, neuron_idx].setub(U_i)
        return {'fixed_on': 0, 'fixed_off': 0, 'unstable': 0}


class ActivationFactory:
    """Factory for creating activation encoders."""
    _encoders = {
        'relu': ReLUEncoder,
        'linear': LinearEncoder,
        'identity': LinearEncoder,
    }
    
    @classmethod
    def create(cls, activation: str) -> ActivationEncoder:
        encoder_class = cls._encoders.get(activation.lower())
        if encoder_class is None:
            raise ValueError(f"Unsupported activation: {activation}")
        return encoder_class()
    
    @classmethod
    def register(cls, name: str, encoder_class: type):
        """Allow users to register custom activations."""
        cls._encoders[name.lower()] = encoder_class


# =============================================================================
# 3. BOUND COMPUTATION STRATEGIES
# =============================================================================

class BoundComputer(ABC):
    """Abstract bound computation strategy."""
    
    @abstractmethod
    def compute_bounds(self, layers: List[LayerInfo], 
                      input_bounds: List[Tuple[float, float]],
                      config: VerifierConfig) -> List[List[Tuple[float, float]]]:
        """Compute bounds for all layers."""
        pass


class IntervalArithmeticComputer(BoundComputer):
    """Fast interval arithmetic bounds."""
    
    def compute_bounds(self, layers: List[LayerInfo],
                      input_bounds: List[Tuple[float, float]],
                      config: VerifierConfig) -> List[List[Tuple[float, float]]]:
        all_bounds = []
        current_bounds = input_bounds
        
        for l_idx, layer in enumerate(layers):
            layer_bounds = self._compute_layer_bounds(
                layer.weights, layer.biases, current_bounds, 
                is_last=(l_idx == len(layers) - 1)
            )
            all_bounds.append(layer_bounds)
            # Apply activation for next layer
            current_bounds = [(max(0, l), max(0, u)) for l, u in layer_bounds]
        
        return all_bounds
    
    def _compute_layer_bounds(self, W, B, prev_bounds, is_last=False):
        bounds = []
        for n_idx in range(len(B)):
            low_sum, high_sum = B[n_idx], B[n_idx]
            for i, w_val in enumerate(W[n_idx]):
                l_prev, h_prev = prev_bounds[i]
                if w_val > 0:
                    low_sum += w_val * l_prev
                    high_sum += w_val * h_prev
                else:
                    low_sum += w_val * h_prev
                    high_sum += w_val * l_prev
            bounds.append((low_sum, high_sum))
        return bounds


class OBBTComputer(BoundComputer):
    """Optimization-based bound tightening."""
    
    def __init__(self, base_model_builder):
        self.base_model_builder = base_model_builder
    
    def compute_bounds(self, layers: List[LayerInfo],
                      input_bounds: List[Tuple[float, float]],
                      config: VerifierConfig) -> List[List[Tuple[float, float]]]:
        if config.verbose:
            print("Starting OBBT...")
        
        ia_computer = IntervalArithmeticComputer()
        ia_bounds = ia_computer.compute_bounds(layers, input_bounds, config)
        
        refined_bounds = []
        for l_idx, layer in enumerate(layers):
            layer_refined = []
            if config.verbose:
                print(f"  Tightening Layer {l_idx}...")
            
            for n_idx in range(layer.num_neurons):
                ia_l, ia_u = ia_bounds[l_idx][n_idx]
                
                # Only optimize unstable neurons
                if ia_l < -config.numerical_tolerance < ia_u:
                    tight_l, tight_u = self._optimize_neuron(
                        layers, l_idx, n_idx, input_bounds, 
                        refined_bounds, ia_bounds, config
                    )
                    layer_refined.append((tight_l, tight_u))
                else:
                    layer_refined.append((ia_l, ia_u))
            
            refined_bounds.append(layer_refined)
        
        return refined_bounds
    
    def _optimize_neuron(self, layers, l_idx, n_idx, input_bounds,
                        refined_bounds, ia_bounds, config):
        """Optimize bounds for a single neuron."""
        # Build model up to current layer with hybrid bounds:
        # - Use refined bounds for previous layers (already computed)
        # - Use IA bounds for current layer (being computed)
        hybrid_bounds = refined_bounds + [ia_bounds[l_idx]]
        
        # Build partial model (only up to current layer)
        model = self.base_model_builder.build(
            layers[:l_idx+1], 
            input_bounds, 
            hybrid_bounds,  # Pass hybrid bounds
            config
        )
        
        # Build target expression for the neuron we're optimizing
        layer = layers[l_idx]
        if l_idx == 0:
            prev_vars = [model.inputs[i] for i in range(len(input_bounds))]
        else:
            prev_vars = [model.y[l_idx-1, i] for i in range(layers[l_idx-1].num_neurons)]
        
        target_expr = sum(
            layer.weights[n_idx][i] * prev_vars[i] 
            for i in range(len(prev_vars))
        ) + layer.biases[n_idx]
        
        # Minimize
        model.obj = aml.Objective(expr=target_expr, sense=aml.minimize)
        solver = SolverFactory(config.solver_name)
        result = solver.solve(model, tee=False)
        
        if result.solver.termination_condition != TerminationCondition.optimal:
            # Fallback to IA bounds if optimization fails
            return ia_bounds[l_idx][n_idx]
        
        tight_l = aml.value(model.obj)
        
        # Maximize (reuse model, just change objective)
        model.del_component(model.obj)
        model.obj = aml.Objective(expr=target_expr, sense=aml.maximize)
        result = solver.solve(model, tee=False)
        
        if result.solver.termination_condition != TerminationCondition.optimal:
            # Fallback to IA bounds if optimization fails
            return ia_bounds[l_idx][n_idx]
        
        tight_u = aml.value(model.obj)
        
        return tight_l, tight_u


class PyomoModelBuilder:
    """Builds Pyomo models for NN encoding."""
    
    def build(self, layers: List[LayerInfo], input_bounds: List[Tuple[float, float]],
              precomputed_bounds: Optional[List[List[Tuple[float, float]]]] = None,
              config: VerifierConfig = None) -> aml.ConcreteModel:
        """Build base NN model."""
        if config is None:
            config = VerifierConfig()
        
        model = aml.ConcreteModel()
        self._add_variables(model, layers, input_bounds, config)
        self._encode_network(model, layers, input_bounds, precomputed_bounds, config)
        return model
    
    def _add_variables(self, model, layers, input_bounds, config):
        """Add variables to model."""
        # Input variables
        model.inputs = aml.Var(range(len(input_bounds)), domain=aml.Reals)
        for i, (low, high) in enumerate(input_bounds):
            model.inputs[i].setlb(low)
            model.inputs[i].setub(high)
        
        # Determine max neurons per layer
        max_neurons = max(layer.num_neurons for layer in layers)
        
        # Neuron output variables
        model.y = aml.Var(range(len(layers)), range(max_neurons), domain=aml.Reals)
        model.z = aml.Var(range(len(layers)), range(max_neurons), domain=aml.Binary)
        model.nn_cons = aml.ConstraintList()
    
    def _encode_network(self, model, layers, input_bounds, precomputed_bounds, config):
        """Encode all layers."""
        stats = {'fixed_on': 0, 'fixed_off': 0, 'unstable': 0}
        
        # Compute bounds if not provided
        if precomputed_bounds is None:
            bound_computer = IntervalArithmeticComputer()
            precomputed_bounds = bound_computer.compute_bounds(layers, input_bounds, config)
        
        # Validate bounds length matches layers
        if len(precomputed_bounds) != len(layers):
            raise ValueError(
                f"Precomputed bounds length ({len(precomputed_bounds)}) "
                f"doesn't match layers length ({len(layers)})"
            )
        
        prev_layer_vars = [model.inputs[i] for i in range(len(input_bounds))]
        
        for l_idx, layer in enumerate(layers):
            # Validate neuron count matches
            if len(precomputed_bounds[l_idx]) != layer.num_neurons:
                raise ValueError(
                    f"Layer {l_idx}: precomputed bounds has {len(precomputed_bounds[l_idx])} "
                    f"neurons but layer has {layer.num_neurons}"
                )
            
            # Determine activation encoder
            is_last_layer = (l_idx == len(layers) - 1)
            activation_type = 'linear' if is_last_layer else layer.activation
            encoder = ActivationFactory.create(activation_type)
            
            # Encode each neuron
            for n_idx in range(layer.num_neurons):
                lin_expr = sum(
                    layer.weights[n_idx][i] * prev_layer_vars[i] 
                    for i in range(len(prev_layer_vars))
                ) + layer.biases[n_idx]
                
                bounds = precomputed_bounds[l_idx][n_idx]
                neuron_stats = encoder.encode(
                    model, l_idx, n_idx, lin_expr, bounds, config
                )
                
                # Aggregate statistics
                for key in stats:
                    stats[key] += neuron_stats[key]
            
            # Update previous layer variables
            prev_layer_vars = [model.y[l_idx, i] for i in range(layer.num_neurons)]
        
        if config.verbose:
            self._print_stats(stats)
    
    def _print_stats(self, stats):
        total = sum(stats.values())
        if total > 0:
            print("-" * 40)
            print("NN ENCODING SUMMARY")
            print(f"Total neurons:   {total}")
            print(f"Fixed ON:        {stats['fixed_on']}")
            print(f"Fixed OFF:       {stats['fixed_off']}")
            print(f"Unstable (Big-M):{stats['unstable']}")
            print(f"Reduction:       {((stats['fixed_on'] + stats['fixed_off']) / total * 100):.1f}%")
            print("-" * 40)



# =============================================================================
# 5. VERIFICATION TASK ABSTRACTION (Template Method Pattern)
# =============================================================================

class VerificationTask(ABC):
    """Abstract base for verification tasks."""
    
    def __init__(self, config: VerifierConfig = None):
        self.config = config or VerifierConfig()
        self.model_builder = PyomoModelBuilder()
    
    def verify(self, layers: List[LayerInfo], input_bounds: List[Tuple[float, float]],
               **kwargs) -> Dict[str, Any]:
        """Template method for verification."""
        # Step 1: Compute bounds
        bounds = self._compute_bounds(layers, input_bounds)
        
        # Step 2: Build base model
        model = self.model_builder.build(layers, input_bounds, bounds, self.config)
        model._spec_ref = kwargs.get('spec')
        
        # Step 3: Add task-specific constraints (hook method)
        self._add_task_constraints(model, layers, **kwargs)
        
        # Step 4: Set objective (hook method)
        self._set_objective(model, layers, **kwargs)
        
        # Step 5: Solve
        return self._solve_and_parse(model, layers)
    
    def _compute_bounds(self, layers, input_bounds):
        """Compute bounds based on config."""
        if self.config.use_obbt:
            computer = OBBTComputer(self.model_builder)
        else:
            computer = IntervalArithmeticComputer()
        return computer.compute_bounds(layers, input_bounds, self.config)
    
    @abstractmethod
    def _add_task_constraints(self, model, layers, **kwargs):
        """Add task-specific constraints."""
        pass
    
    @abstractmethod
    def _set_objective(self, model, layers, **kwargs):
        """Set optimization objective."""
        pass
    
    def _solve_and_parse(self, model, layers):
        """Solve model and parse results."""
        opt = SolverFactory(self.config.solver_name)
        results = opt.solve(model, tee=self.config.verbose, 
                          options=self.config.solver_options)
        
        if results.solver.termination_condition != TerminationCondition.optimal:
            return {"status": str(results.solver.termination_condition)}
        
        return self._parse_solution(model, layers)
    
    @abstractmethod
    def _parse_solution(self, model, layers) -> Dict[str, Any]:
        """Parse solution into results dictionary."""
        pass


from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import yaml
import numpy as np


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

class ConfigParser:
    """Parse YAML configuration for NN verification."""
    
    @staticmethod
    def load_from_yaml(filepath: str) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        return ConfigParser.parse_config(config)
    
    @staticmethod
    def parse_config(config: dict) -> Dict[str, Any]:
        """
        Parse configuration dictionary into structured components.
        
        Returns:
            Dict with keys: 'layers', 'verification_spec', 'model_meta'
        """
        # Parse layers
        layers = ConfigParser._parse_layers(config['layers'])
        
        # Parse verification spec
        verification_spec = ConfigParser._parse_verification_spec(
            config['verification_spec']
        )
        
        # Parse model metadata
        model_meta = config.get('model_meta', {})
        
        return {
            'layers': layers,
            'verification_spec': verification_spec,
            'model_meta': model_meta
        }
    
    @staticmethod
    def _parse_layers(layers_config: List[dict]) -> List[LayerInfo]:
        """Parse layers from config."""
        layers = []
        for i, layer_dict in enumerate(layers_config):
            # Determine activation (all but last have ReLU by default)
            is_last = (i == len(layers_config) - 1)
            activation = layer_dict.get('activation', 'linear' if is_last else 'relu')
            
            layers.append(LayerInfo(
                weights=np.array(layer_dict['weights']),
                biases=np.array(layer_dict['biases']),
                activation=activation
            ))
        return layers
    
    @staticmethod
    def _parse_verification_spec(spec_config: dict) -> VerificationSpec:
        """Parse agnostic verification specification with index mapping."""
        
        # 1. Parse input bounds (remains mostly same)
        input_bounds = [
            (float(bound['min']), float(bound['max'])) 
            for bound in spec_config['input_bounds']
        ]
        
        # 2. Extract Mapping Indices
        # This tells the verifier which parts of the physics vector 'x' the NN handles
        mapping = spec_config.get('indices', {})
        input_indices = mapping.get('input_indices', [])
        output_indices = mapping.get('output_indices', [])
        
        # 3. Parse Physics Parameters
        constraints = spec_config.get('constraints', {})
        A = np.array(constraints.get('A', []))
        b_static = np.array(constraints.get('b_static', []))
        
        # 4. Parse Objective
        c = np.array(spec_config.get('objective_c', []))
        
        return VerificationSpec(
            input_bounds=input_bounds,
            constraints_A=A,
            objective_c=c,
            b_static=b_static,
            input_indices=input_indices,
            output_indices=output_indices
        )


# =============================================================================
# BACKWARD-COMPATIBLE RESULT WRAPPER
# =============================================================================

class VerificationResult(dict):
    """
    Result dictionary with backward compatibility for 'val' key.
    
    This allows both old code (result['val']) and new code 
    (result['max_violation']) to work.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up backward compatibility
        self._setup_aliases()
    
    def _setup_aliases(self):
        """Create aliases for backward compatibility."""
        # Map new keys to old 'val' key
        if 'max_violation' in self:
            self['val'] = self['max_violation']
        elif 'max_gap' in self:
            self['val'] = self['max_gap']
        elif 'optimality_gap' in self:
            self['val'] = self['optimality_gap']
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key in ['max_violation', 'max_gap', 'optimality_gap']:
            super().__setitem__('val', value)


# =============================================================================
# UPDATE VERIFICATION TASKS TO USE VerificationResult
# =============================================================================

class FeasibilityVerificationTask(VerificationTask):
    """Verify LP feasibility using agnostic index mapping."""
    
    def _add_task_constraints(self, model, layers, **kwargs):
        # 1. Extract the new spec data
        spec = kwargs['spec'] # The VerificationSpec object
        A = spec.constraints_A
        b_static = spec.b_static
        in_idx = spec.input_indices
        out_idx = spec.output_indices
        
        model._A_matrix = spec.constraints_A
        model._b_vector = spec.b_static
        model._out_idx = spec.output_indices
        
        output_layer_idx = len(layers) - 1
        num_vars = A.shape[1]  # Total variables in the physics problem
        
        # 2. Create the "Global x" variables (x_full)
        # This vector represents the full state of the physical system
        model.x_full = aml.Var(range(num_vars), domain=aml.Reals, bounds=(-1e6, 1e6))
        
        # 3. LINKING STEP: Stitch the NN into the Global x vector
        # Link Input Variables: x_full[in_idx] == model.inputs
        for i, global_idx in enumerate(in_idx):
            model.nn_cons.add(model.x_full[global_idx] == model.inputs[i])
            
        # Link Output Variables: x_full[out_idx] == model.y[last_layer]
        for i, global_idx in enumerate(out_idx):
            model.nn_cons.add(model.x_full[global_idx] == model.y[output_layer_idx, i])

        # 4. Max violation variable
        model.max_viol = aml.Var(domain=aml.Reals, bounds=(-self.config.bound_u, self.config.bound_u))
        
        # 5. Physics Check: A * x_full <= b_static
        for j in range(len(model._A_matrix)):
            row_sum = sum(model._A_matrix[j][i] * model.x_full[i] for i in range(num_vars))
            model.nn_cons.add(model.max_viol >= row_sum - model._b_vector[j])
    
    def _set_objective(self, model, layers, **kwargs):
        model.obj = aml.Objective(expr=model.max_viol, sense=aml.maximize)
    
    def _parse_solution(self, model, layers):
        out_layer_idx = len(layers) - 1
        A = model._A_matrix 
        b = model._b_vector
        
        # Standardize values to float to avoid numpy/pyomo type issues
        full_x = [float(aml.value(model.x_full[i])) for i in range(len(model.x_full))]
        
        # Calculate violations (Ax - b)
        violations = [float(sum(A[j][i] * full_x[i] for i in range(len(full_x))) - b[j]) for j in range(len(b))]
        max_violation = float(max(violations))

        # Construct the dictionary with EVERY key the system might look for
        result = {
            "status": "Success",
            "max_violation": max_violation,
            "optimality_gap": max_violation, # Aliasing for compatibility
            "true_optimal_cost": 0.0,         # Placeholder for constraint checks
            "violation_per_row": violations,
            "at_input_val": [float(aml.value(model.inputs[i])) for i in range(len(model.inputs))],
            "nn_vals": [float(aml.value(model.y[out_layer_idx, i])) for i in range(layers[-1].num_neurons)],
            "opt_vals": [0.0] * layers[-1].num_neurons,
            "full_x_vector": full_x  
        }
        
        return result


class OptimalityGapVerificationTask(VerificationTask):
    """Verify LP optimality gap for agnostic partial mappings."""
    
    def _add_task_constraints(self, model, layers, **kwargs):
        spec = kwargs['spec']
        A, b, c = spec.constraints_A, spec.b_static, spec.objective_c
        in_idx, out_idx = spec.input_indices, spec.output_indices
        m, n = A.shape
        BIG_M = self.config.big_m
        
        # Park metadata on the model for later parsing
        model._objective_c = spec.objective_c
        model._output_indices = spec.output_indices
        
        # 1. Variables
        model.x_star = aml.Var(range(n), domain=aml.Reals, bounds=(0, 1e4))
        model.x_nn_full = aml.Var(range(n), domain=aml.Reals, bounds=(0, 1e4))
        model.lmbda = aml.Var(range(m), domain=aml.NonNegativeReals, bounds=(0, 1e8))
        model.kkt_select = aml.Var(range(m), domain=aml.Binary)

        # 2. Input Linking: THIS IS VITAL
        # Both states must use the verifier's chosen "worst-case" input
        for i, g_idx in enumerate(in_idx):
            model.nn_cons.add(model.x_star[g_idx] == model.inputs[i])
            model.nn_cons.add(model.x_nn_full[g_idx] == model.inputs[i])

        # 3. Stationarity: ONLY for the degrees of freedom (Outputs)
        # This matches exactly what scipy.linprog does when you fix inputs.
        for i in out_idx:
            dual_lhs = sum(A[j][i] * model.lmbda[j] for j in range(m))
            # If minimizing positive cost:
            model.nn_cons.add(dual_lhs <= c[i] + 1e-7) 

        # 4. Primal Feasibility & Complementary Slackness
        # This links the duals (lambda) to the rows (b - Ax)
        for j in range(m):
            row_expr_star = sum(A[j][i] * model.x_star[i] for i in range(n))
            
            # Ax <= b
            model.nn_cons.add(row_expr_star <= b[j] + 1e-7)
            
            # If lambda > 0, then Ax == b (Constraint is active)
            # If b - Ax > 0, then lambda == 0 (Constraint is inactive)
            model.nn_cons.add(b[j] - row_expr_star <= BIG_M * model.kkt_select[j])
            model.nn_cons.add(model.lmbda[j] <= BIG_M * (1 - model.kkt_select[j]))

        # 5. NN Output Linking
        out_layer_idx = len(layers) - 1
        for i, g_idx in enumerate(out_idx):
            model.nn_cons.add(model.x_nn_full[g_idx] == model.y[out_layer_idx, i])

    def _set_objective(self, model, layers, **kwargs):
        spec = kwargs['spec']
        c = spec.objective_c
        # Regret = NN_Cost - Optimal_Cost
        nn_cost = sum(c[i] * model.x_nn_full[i] for i in range(len(c)))
        true_cost = sum(c[i] * model.x_star[i] for i in range(len(c)))
        model.obj = aml.Objective(expr=nn_cost - true_cost, sense=aml.maximize)
    
    def _parse_solution(self, model, layers):
        out_layer_idx = len(layers) - 1
        
        # Retrieve the parked metadata
        c = model._objective_c
        out_idx_list = model._output_indices

        # Calculate costs
        nn_total_cost = sum(c[i] * aml.value(model.x_nn_full[i]) for i in range(len(c)))
        optimal_total_cost = sum(c[i] * aml.value(model.x_star[i]) for i in range(len(c)))

        # Construct the result dictionary
        result = {
            "status": "Success",
            "optimality_gap": aml.value(model.obj),
            "true_optimal_cost": optimal_total_cost,
            "nn_total_cost": nn_total_cost,
            "at_input_val": [aml.value(model.inputs[i]) for i in range(len(model.inputs))],
            "nn_vals": [aml.value(model.y[out_layer_idx, i]) for i in range(layers[-1].num_neurons)],
            "opt_vals": [aml.value(model.x_star[i]) for i in out_idx_list],
            "x_star": [aml.value(model.x_star[i]) for i in range(len(model.x_star))],
            "full_x_vector": [aml.value(model.x_nn_full[i]) for i in range(len(model.x_nn_full))]
        }
        
        # Return as the VerificationResult object
        return VerificationResult(result)
    


# =============================================================================
# 6. MAIN VERIFIER CLASS (Facade Pattern)
# =============================================================================

class MILPVerifier:
    """
    Unified interface for neural network verification.
    Supports multiple verification tasks and is easily extensible.
    """
    
    def __init__(self, layer_params: List[dict], config: VerifierConfig = None):
        """
        Args:
            layer_params: List of dicts with 'weights', 'biases', and optionally 'activation'
            config: Verification configuration
        """
        self.config = config or VerifierConfig()
        self.layers = [
            LayerInfo(
                weights=np.array(lp['weights']),
                biases=np.array(lp['biases']),
                activation=lp.get('activation', 'relu')
            )
            for lp in layer_params
        ]
    
    def verify_lp_feasibility(self, spec: VerificationSpec, input_bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Verify LP feasibility (maximum constraint violation)."""
        task = FeasibilityVerificationTask(self.config)
        # Call the verify method of the task instance
        return task.verify(
            layers=self.layers, 
            input_bounds=input_bounds, 
            spec=spec
        )
    
    def verify_lp_optimality_gap(self, spec: VerificationSpec, input_bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Verify LP optimality gap using KKT conditions."""
        task = OptimalityGapVerificationTask(self.config)
        # Call the verify method of the task instance
        return task.verify(
            layers=self.layers, 
            input_bounds=input_bounds, 
            spec=spec
        )
    
    def add_verification_task(self, task_name: str, task_class: type):
        """Register custom verification task."""
        setattr(self, f"verify_{task_name}", 
                lambda **kwargs: task_class(self.config).verify(self.layers, **kwargs))






























# import pyomo.environ as aml
# from pyomo.opt import SolverFactory, TerminationCondition
# import numpy as np

# class MILPVerifier:
#     def __init__(self, layer_params, activation='relu', big_m=1e6, bound_u=1000.0):
#         self.layers = layer_params
#         self.activation = activation.lower()
#         self.M = big_m
#         self.U = bound_u

#     def _setup_base_model(self, input_bounds, precomputed_bounds=None):
#         model = aml.ConcreteModel()
        
#         # 1. Define Input Variables
#         model.inputs = aml.Var(range(len(input_bounds)), domain=aml.Reals)
#         for i, (low, high) in enumerate(input_bounds):
#             model.inputs[i].setlb(low)
#             model.inputs[i].setub(high)

#         model.y = aml.Var(range(len(self.layers)), range(1000), domain=aml.Reals)
#         model.z = aml.Var(range(len(self.layers)), range(1000), domain=aml.Binary)
#         model.nn_cons = aml.ConstraintList()

#         # Summary Counters
#         total_neurons = 0
#         fixed_on = 0
#         fixed_off = 0
#         unstable = 0

#         current_ia_bounds = input_bounds 
#         prev_layer_vars = [model.inputs[i] for i in range(len(input_bounds))]
        
#         for l_idx, layer in enumerate(self.layers):
#             W, B = layer['weights'], layer['biases']
            
#             if precomputed_bounds and l_idx < len(precomputed_bounds):
#                 layer_bounds = precomputed_bounds[l_idx]
#             else:
#                 layer_bounds, _ = self._interval_arithmetic(W, B, current_ia_bounds, l_idx)
            
#             for n_idx in range(len(W)):
#                 lin_expr = sum(W[n_idx][i] * prev_layer_vars[i] for i in range(len(prev_layer_vars))) + B[n_idx]
#                 L_i, U_i = layer_bounds[n_idx]
#                 total_neurons += 1

#                 if l_idx < len(self.layers) - 1:
#                     # Apply ReLU Pruning/Fixing
#                     if L_i >= 0:
#                         # Case A: Always ON
#                         model.nn_cons.add(model.y[l_idx, n_idx] == lin_expr)
#                         model.z[l_idx, n_idx].fix(1)
#                         model.y[l_idx, n_idx].setlb(L_i)
#                         model.y[l_idx, n_idx].setub(U_i)
#                         fixed_on += 1
#                     elif U_i <= 0:
#                         # Case B: Always OFF
#                         model.nn_cons.add(model.y[l_idx, n_idx] == 0)
#                         model.z[l_idx, n_idx].fix(0)
#                         model.y[l_idx, n_idx].setlb(0)
#                         model.y[l_idx, n_idx].setub(0)
#                         fixed_off += 1
#                     else:
#                         # Case C: UNSTABLE (Requires Big-M)
#                         model.nn_cons.add(model.y[l_idx, n_idx] >= lin_expr)
#                         model.nn_cons.add(model.y[l_idx, n_idx] >= 0)
#                         model.nn_cons.add(model.y[l_idx, n_idx] <= lin_expr + abs(L_i) * (1 - model.z[l_idx, n_idx]))
#                         model.nn_cons.add(model.y[l_idx, n_idx] <= U_i * model.z[l_idx, n_idx])
#                         model.y[l_idx, n_idx].setlb(0)
#                         model.y[l_idx, n_idx].setub(U_i)
#                         unstable += 1
#                 else:
#                     # Output Layer
#                     model.nn_cons.add(model.y[l_idx, n_idx] == lin_expr)
#                     model.y[l_idx, n_idx].setlb(L_i)
#                     model.y[l_idx, n_idx].setub(U_i)

#             prev_layer_vars = [model.y[l_idx, i] for i in range(len(W))]
#             if precomputed_bounds:
#                 current_ia_bounds = [(max(0, l), max(0, u)) for l, u in layer_bounds]
#             else:
#                 _, current_ia_bounds = self._interval_arithmetic(W, B, current_ia_bounds, l_idx)

#         # # Print the Presolve Summary
#         # print("-" * 30)
#         # print("MILP PRESOLVE SUMMARY")
#         # print(f"Total ReLU Neurons: {total_neurons - len(W)}") # Exclude output layer
#         # print(f"Fixed ALWAYS ON:    {fixed_on}")
#         # print(f"Fixed ALWAYS OFF:   {fixed_off}")
#         # print(f"Unstable (Big-M):   {unstable}")
#         # print(f"Reduction:          {((fixed_on + fixed_off) / (total_neurons - len(W)) * 100):.1f}%")
#         # print("-" * 30)

#         return model

#     def _interval_arithmetic(self, W, B, prev_bounds, l_idx):
#         """
#         Calculates pre-activation (linear) and post-activation (ReLU) bounds.
#         """
#         lin_bounds = []
#         act_bounds = []
        
#         for n_idx in range(len(W)):
#             low_sum = B[n_idx]
#             high_sum = B[n_idx]
            
#             for i, w_val in enumerate(W[n_idx]):
#                 l_prev, h_prev = prev_bounds[i]
#                 if w_val > 0:
#                     low_sum += w_val * l_prev
#                     high_sum += w_val * h_prev
#                 else:
#                     low_sum += w_val * h_prev
#                     high_sum += w_val * l_prev
            
#             lin_bounds.append((low_sum, high_sum))
            
#             # ReLU logic for the next layer's input
#             if l_idx < len(self.layers) - 1:
#                 act_bounds.append((max(0, low_sum), max(0, high_sum)))
#             else:
#                 # Last layer usually doesn't have ReLU
#                 act_bounds.append((low_sum, high_sum))
                
#         return lin_bounds, act_bounds
    
#     def compute_obbt_bounds(self, input_bounds, solver_name='glpk'):
#         """
#         Layer-by-layer Optimization Based Bound Tightening.
#         """
#         print("Starting OBBT (this may take a moment)...")
#         refined_bounds = []
#         current_input_bounds = input_bounds

#         for l_idx in range(len(self.layers)):
#             layer_refined = []
#             #print(f" Tightening Layer {l_idx}...")
            
#             # 1. Get IA bounds first as a baseline
#             W, B = self.layers[l_idx]['weights'], self.layers[l_idx]['biases']
#             lin_ia, _ = self._interval_arithmetic(W, B, current_input_bounds, l_idx)

#             for n_idx in range(len(W)):
#                 ia_l, ia_u = lin_ia[n_idx]

#                 # 2. Only run OBBT if neuron is unstable (crosses zero)
#                 if ia_l < 0 < ia_u:
#                     # We build a model up to this layer only
#                     # We pass the bounds we've refined so far
#                     partial_model = self._setup_base_model(input_bounds, precomputed_bounds=refined_bounds)
                    
#                     # Target neuron expression
#                     prev_vars = [partial_model.inputs[i] for i in range(len(input_bounds))] if l_idx == 0 else \
#                                 [partial_model.y[l_idx-1, i] for i in range(len(self.layers[l_idx-1]['weights']))]
                    
#                     target_expr = sum(W[n_idx][i] * prev_vars[i] for i in range(len(prev_vars))) + B[n_idx]
                    
#                     # Solve Min
#                     partial_model.obj = aml.Objective(expr=target_expr, sense=aml.minimize)
#                     SolverFactory(solver_name).solve(partial_model)
#                     tight_l = aml.value(partial_model.obj)
                    
#                     # Solve Max (Reuse model, just change objective)
#                     partial_model.del_component(partial_model.obj)
#                     partial_model.obj = aml.Objective(expr=target_expr, sense=aml.maximize)
#                     SolverFactory(solver_name).solve(partial_model)
#                     tight_u = aml.value(partial_model.obj)
                    
#                     layer_refined.append((tight_l, tight_u))
#                     #print(f"Neuron {n_idx}: IA [{ia_l:.2f}, {ia_u:.2f}] -> OBBT [{tight_l:.2f}, {tight_u:.2f}]")
#                 else:
#                     # If IA already proved it's stable, don't waste time solving
#                     layer_refined.append((ia_l, ia_u))
            
#             refined_bounds.append(layer_refined)
#             # ReLU for next layer input
#             current_input_bounds = [(max(0, l), max(0, u)) for l, u in layer_refined]

#         return refined_bounds

#     def verify_lp_feasibility(self, A, input_bounds, solver_name='glpk', use_obbt = True):
#         """
#         Calculates the global maximum constraint violation for the NN surrogate.
        
#         Logic:
#         Instead of checking constraints one by one, we linearize the 'max' function. 
#         We introduce a helper variable 'max_viol' and constrain it to be greater 
#         than or equal to every individual row violation (Ax - b). 
        
#         By maximizing 'max_viol', the solver finds the input 'b' that pushes 
#         the Neural Network's prediction 'x' as far as possible outside the 
#         feasible region of ANY constraint.
        
#         Returns:
#             Dict: Results containing the worst violation value and the 
#                   counter-example input 'b' that caused it.
#         """
        
#         refined = None
#         if use_obbt:
#             # This uses the method you provided to calculate tighter L_i, U_i
#             refined = self.compute_obbt_bounds(input_bounds, solver_name=solver_name)
            
#         # 1. Initialize the base Pyomo model with NN Big-M logic
#         model = self._setup_base_model(input_bounds, precomputed_bounds=refined)
#         output_layer_idx = len(self.layers) - 1
#         num_outputs = len(self.layers[-1]['biases'])

#         # 2. Define the 'max_viol' helper variable
#         # This variable acts as an upper envelope for all individual row violations.
#         model.max_viol = aml.Var(domain=aml.Reals, bounds=(-self.U, self.U))
        
#         # 3. Add Linearized Max Constraints
#         # For each row j: max_viol >= (A[j] * x) - b[j]
#         for j in range(len(A)):
#             # Calculate the specific violation for this constraint row
#             row_viol = sum(A[j][i] * model.y[output_layer_idx, i] for i in range(num_outputs)) - model.inputs[j]
            
#             # Constraint: max_viol must be at least as large as this specific violation
#             model.nn_cons.add(model.max_viol >= row_viol)
        
#         # 4. Objective: Maximize the helper variable to find the absolute worst case
#         model.obj = aml.Objective(expr=model.max_viol, sense=aml.maximize)
        
#         return self._solve(model, solver_name, is_feasibility=True)

#     def verify_lp_optimality_gap(self, spec, input_bounds, solver_name='glpk'):
#         """
#         Finds the absolute worst-case gap between NN and True Optimal.
#         Uses Big-M to linearize KKT conditions for GLPK.
#         """
#         # 1. Setup Model (Includes NN logic)
#         refined_bounds = self.compute_obbt_bounds(input_bounds, solver_name=solver_name)
#         model = self._setup_base_model(input_bounds, precomputed_bounds=refined_bounds)
        
#         out_idx = len(self.layers) - 1
#         A = np.array(spec["A"])
#         c = np.array(spec["c"])
#         m, n = A.shape  # m constraints, n vars
#         M = self.M

#         # 2. Variables for the "True Optimal" solution (x_star) and Duals (lmbda)
#         model.x_star = aml.Var(range(n), domain=aml.NonNegativeReals, bounds=(0, 10))
#         model.lmbda = aml.Var(range(m), domain=aml.NonNegativeReals, bounds=(0, 10))
#         model.kkt_select = aml.Var(range(m), domain=aml.Binary) # Binary for slackness

#         # 3. KKT Constraints (The "Truth" embedding)
#         for j in range(m):
#             # Primal Feasibility: Ax <= b
#             row_expr = sum(A[j][i] * model.x_star[i] for i in range(n))
#             model.nn_cons.add(row_expr <= model.inputs[j])
            
#             # Complementary Slackness (Big-M version)
#             # If kkt_select=0: Slack is 0 (Ax = b)
#             # If kkt_select=1: Lambda is 0
#             model.nn_cons.add(model.inputs[j] - row_expr <= M * model.kkt_select[j])
#             model.nn_cons.add(model.lmbda[j] <= M * (1 - model.kkt_select[j]))

#         for i in range(n):
#             # Dual Feasibility: A.T * lambda >= c (since we minimize c^T x)
#             # Note: For min c^T x, dual is A^T * lambda <= c if using different sign convention
#             dual_lhs = sum(A[j][i] * model.lmbda[j] for j in range(m))
#             model.nn_cons.add(dual_lhs >= c[i])

#         # 4. Objective: Maximize (NN_Cost - True_Optimal_Cost)
#         nn_cost = sum(c[i] * model.y[out_idx, i] for i in range(n))
#         true_cost = sum(c[i] * model.x_star[i] for i in range(n))
        
#         model.obj = aml.Objective(expr=nn_cost - true_cost, sense=aml.maximize)

#         return self._solve(model, solver_name, is_feasibility=False)

#     def _solve(self, model, solver_name, is_feasibility=False):
#         """INTERNAL: Handles the solver call and result parsing."""
#         opt = SolverFactory(solver_name)
#         results = opt.solve(model, tee=False)

#         if results.solver.termination_condition == TerminationCondition.optimal:
#             out_idx = len(self.layers) - 1
#             num_outputs = len(self.layers[out_idx]['biases'])
            
#             # Extract the actual values from the model
#             input_values = [aml.value(model.inputs[i]) for i in range(len(model.inputs))]
#             output_values = [aml.value(model.y[out_idx, i]) for i in range(num_outputs)]
            
#             res = {
#                 "status": "Success",
#                 "val": aml.value(model.obj),
#                 "at_input_b": input_values,
#                 "at_output_x": output_values  # Added explicitly here
#             }
            
#             if hasattr(model, 'x_star'):
#                 res["x_star"] = [aml.value(model.x_star[i]) for i in range(len(model.x_star))]
            
#             # Friendly naming based on context while keeping the raw keys
#             if is_feasibility:
#                 res["max_violation"] = res["val"]
#             else:
#                 res["max_predicted_cost"] = res["val"]
                
#             return res
            
#         return {"status": str(results.solver.termination_condition)}
