#implementation was made from offical github https://github.com/mlech26l/ncps/tree/master/ncps but accomodated to our understanding
import torch 
import torch.nn as nn
import numpy as np 

#---------------------------
# Liquid Time-Constant Cell - Defines a forward pass for a single time step utilising the NCP neural wiring; any adjustments to forward pass gets modified here
#---------------------------
class LTCCell(nn.Module):
    def __init__(
        self, 
        wiring, 
        input_mapping="affine", #or linear; both mappings do the same thing but affine allows enables a bias parameter to be part of the network
        output_mapping="affine", 
        ode_unfolds=6, 
        epsilon=1e-8,
        implicit_constraints=False):

        super(LTCCell, self).__init__() #inherent from pytorch 

        if not wiring.is_built():
            raise ValueError(f"error initialising ncp neuron wiring, please call wiring.build(input_dim) first")
        
        #function to enforce positive parameter values: membrane capacitance, self.params['w'] and self.params['sensory_w'] and leakage conductance 
        self.make_positive = nn.Softplus() if implicit_constraints else nn.Identity() #makes certain values positive using softplus otherwise leave as is using identity

        self.init_ranges = {
            #internal neuron parameters
            "leakage_conductance": (0.001, 1.0), #controls leak current
            "reverse_potential": (-0.2, 0.2),   #internal neuron reverse potential
            "membrane_capacitance": (0.4, 0.6),  #time constant scaling

            #synaptic parameters
            "w": (0.001, 1.0), 
            "sigma": (3, 8),  
            "mu": (0.3, 0.8),

            #sensory synaptic parameters     
            "sensory_w": (0.001, 1.0),     
            "sensory_sigma": (3, 8),              
            "sensory_mu": (0.3, 0.8)           
        }

        #store configuration
        self.wiring = wiring
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.ode_unfolds = ode_unfolds  #how many times to approximate hidden state evolution per single time step process, by default every time step has 6 unfolds
        self.epsilon = epsilon  #small constant to prevent division by zero

        #synaptic weight capture for visualisation
        self.capture_synaptic_weights = False
        self.last_synaptic_weights = None

        #initialise parameters
        self.initialise_parameters()

#---------------------------class properties---------------------------
    @property
    def total_neurons(self):
        return self.wiring.get_total_neurons(exclude_sensory=False)

    @property 
    def internal_neuron_size(self):
        return self.wiring.get_total_neurons(exclude_sensory=True)
    
    @property
    def sensory_size(self):
        return self.wiring.input_dim 
    
    @property
    def motor_size(self):
        return self.wiring.output_dim
    
    @property 
    def synpase_count(self):
        return self.wiring.synapse_count 
    
    @property
    def sensory_synapse_count(self):
        return self.wiring.sensory_synapse_count

#---------------------------neurons/parameter initialisation---------------------------
    #initialising parameter types with weight values    
    def add_weight(self, name, init_value, requires_grad=True):
        param = nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param) #store the parameter with a name to keep track i.e sensory weights
        return param
    
    #helper function used to initialise parameters with min max values from init_ranges dictionary
    def get_init_value(self, shape, param_name):
        minval, maxval = self.init_ranges[param_name] 

        if minval == maxval:
            return torch.ones(shape) * minval 
        else:
            return torch.rand(*shape) * (maxval - minval) + minval #*shape passes shape iterable and unpacks every element

    #initialise all parameters
    def initialise_parameters(self):
        self.params = {}

        #internal neuron parameters
        keys = ["leakage_conductance", "reverse_potential", "membrane_capacitance"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.internal_neuron_size,), name))
        
        #neuron to neuron connection parameters
        keys = ["w", "sigma", "mu"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.internal_neuron_size, self.internal_neuron_size), name))
        
        #note that neuron and sensory reverse potentials are already initialised with values
        #neuron to neuron reverse potential
        self.params['synapse_reverse_potential'] = self.add_weight(name="synapse_reverse_potential", init_value=torch.Tensor(self.wiring.create_synapse_reverse_potential(self.wiring.get_NAM)))

        #sensory to neuron connection parameters
        keys = ["sensory_w", "sensory_sigma", "sensory_mu"]
        for name in keys:
            self.params[name] = self.add_weight(name=name, init_value=self.get_init_value((self.sensory_size, self.internal_neuron_size), name))
        
        #sensory reverse potential
        self.params["sensory_reverse_potential"] = self.add_weight(name="sensory_reverse_potential", init_value=torch.Tensor(self.wiring.create_synapse_reverse_potential(self.wiring.get_SAM))) 

        #sparsity masks, they are non-trainable
        keys = ["sparsity_mask", "sensory_sparsity_mask"]
        self.params[keys[0]] = self.add_weight(name=keys[0], init_value=torch.Tensor(np.abs(self.wiring.get_NAM)), requires_grad=False)
        self.params[keys[1]] = self.add_weight(name=keys[1], init_value=torch.Tensor(np.abs(self.wiring.get_SAM)), requires_grad=False)

        #optional input and output mappings
        if self.input_mapping in ["affine", "linear"]:
            self.params["input_weights"] = self.add_weight(name="input_weights", init_value=torch.ones((self.sensory_size,)))

        if self.input_mapping == "affine":
            self.params["input_bias"] = self.add_weight(name="input_bias", init_value=torch.zeros((self.sensory_size,)))
        
        if self.output_mapping in ["affine", "linear"]:
            self.params["output_weights"] = self.add_weight(name="output_weights", init_value=torch.ones((self.motor_size,)))

        if self.output_mapping == "affine":
            self.params["output_bias"] = self.add_weight(name="output_bias", init_value=torch.zeros((self.motor_size,)))

#---------------------------forward pass/layers---------------------------
    def forward(self, x, state, time_constant): #single time step, so x is shape batch x feature_dim
        #handle initial state, expect recurrent loop to be outside this class to passing recurrent neural states, this cell is defined for a single timestep
        if state is None:
            batch_size = x.size(0)
            state = torch.zeros(batch_size, self.internal_neuron_size, 
                            device=x.device, dtype=x.dtype)
        
        #map inputs
        x_transformed = self.map_input(x)
        new_hidden_state = self.fused_solver(x_transformed, state, time_constant) #evolve hidden state
        outputs = self.map_outputs(new_hidden_state) #get outputs but theyre unused in eeg context, we use the hidden state for the output
        
        return outputs, new_hidden_state

    #ODE Solver - computes neural evolution, where hidden state (the internal neurons) evolve according to neural ODE
    def fused_solver(self, x_transformed, state, time_constant):
        current_state = state 

        #compute sensory synaptic activation and reverse potential
        sensory_numerator, sensory_denominator = self.compute_synapse(input=x_transformed, 
                                                                      weight=self.make_positive(self.params['sensory_w']), #enforce positive if using implicit constraint
                                                                      mu=self.params['sensory_mu'],
                                                                      sigma=self.params['sensory_sigma'],
                                                                      sparsity_mask=self.params['sensory_sparsity_mask'],
                                                                      reverse_potential=self.params['sensory_reverse_potential'])
        
        scaled_capacitance = self.make_positive(self.params["membrane_capacitance"]) / (time_constant / self.ode_unfolds) #enforce positive if using implicit constraint

        #ode unfolds
        for _ in range(self.ode_unfolds):
            #compute neuron synaptic activation and reverse potential
            numerator, denominator = self.compute_synapse(input=current_state,
                                                          weight=self.make_positive(self.params["w"]), #enforce positive if using implicit constraint
                                                          mu=self.params['mu'],
                                                          sigma=self.params['sigma'],
                                                          sparsity_mask=self.params['sparsity_mask'],
                                                          reverse_potential=self.params['synapse_reverse_potential'])
            
            synaptic_numerator = numerator + sensory_numerator
            synaptic_denominator = denominator + sensory_denominator

            #neuron parameters broadcasted on synatpic weights, ODE solver
            dh = scaled_capacitance * current_state + self.make_positive(self.params['leakage_conductance']) * self.params['reverse_potential'] + synaptic_numerator
            dt = scaled_capacitance + self.make_positive(self.params['leakage_conductance']) + synaptic_denominator 

            current_state = dh / (dt + self.epsilon) #approximate new state 
        
        return current_state

    #compute the synpatic weights - the intuition here is you use fixed parameters to construct an non-fixed synaptic weight between neurons, this is where the liquid part comes with changing 'weights'
    def compute_synapse(self, input, weight, mu, sigma, sparsity_mask, reverse_potential):
        synaptic_activation = (weight * self.sigmoid_gate(input, mu, sigma)) * sparsity_mask 

        #capture synaptic weights if enabled and single batch
        if self.capture_synaptic_weights and input.shape[0] == 1:
            self.last_synaptic_weights = synaptic_activation.detach().cpu().numpy()[0]

        synaptic_reverse_potential = synaptic_activation * reverse_potential

        return torch.sum(synaptic_reverse_potential, dim=1), torch.sum(synaptic_activation, dim=1) #numerator, denominator respectively

    #gating function to process input with synaptic weight calculation
    def sigmoid_gate(self, x, mu, sigma): #used in both sensory gating and synaptic gating
        x = torch.unsqueeze(x, -1)
        mu = x - mu
        x = sigma * mu 
        return torch.sigmoid(x)
    
    def map_input(self, x):
        if self.input_mapping in ["affine", 'linear']: #both affine and linear perform linear transformation, affine provides the additional bias parameter calculation
            x = x * self.params['input_weights']
        
        if self.input_mapping == "affine":
            x = x + self.params['input_bias']
        
        return x

    def map_outputs(self, state):
        output = state 
        if self.motor_size < self.internal_neuron_size:
            output = output[:, 0:self.motor_size] #output is sliced to motor amount of neurons
        
        if self.output_mapping in ["affine", 'linear']:
            output = output * self.params["output_weights"]
        if self.output_mapping == "affine":
            output = output + self.params['output_bias']
        
        return output

    #enable or disable synaptic weight capture for visualisation
    def set_synaptic_weight_capture(self, enabled=True):
        self.capture_synaptic_weights = enabled

    #return the last captured synaptic weights
    def get_synaptic_weights(self):
        return self.last_synaptic_weights