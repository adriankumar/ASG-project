#This class defines the neural wiring connectivity. 
#We specifically implement the NeuralCircuitPolicy (NCP) wiring which defines 'inter', 'command' and 'motor' neuron types; alternative options are dense connectivity and random sparse connectivity which are not implemented here
#implementation was made from offical github https://github.com/mlech26l/ncps/tree/master/ncps but accomodated to our understanding
import numpy as np

#---------------------------
# Base neuron synapse wiring class - builds empty adjacency matrices to store connections/synapses between neurons with helper functions; Using in NCP class
#---------------------------
class NeuronSynapseWiring:
    def __init__(self, internal_total_neurons):
        self.internal_neuron_total = internal_total_neurons #internal neurons is the sum of inter, command and motor neurons the model uses
        self.neuron_adjacency_matrix = np.zeros([self.internal_neuron_total, self.internal_neuron_total], dtype=np.int8) #inter, command and motor neurons adjacency matrix; dtype=np.int8 for memory effiency; adjacency matrices only store -1, 1 or 0 (polarity)
        self.sensory_adjacency_matrix = None #will be initialised; stores connectivity between sensory neurons and inter neurons

        self.input_dim, self.output_dim = None, None #dimensions of input (features) and output (classes)

    #check if wiring is built ~ both adjacency matrices are initialised
    def is_built(self):
        return self.input_dim is not None

    #get number of neurons in wiring; use exclude_sensory=True to get internal neuron count
    def get_total_neurons(self, exclude_sensory=False):
        if not self.is_built():
            raise ValueError(f"Wiring has no sensory neurons, initialise wiring with input dimension using build() method ~ NeuronWiring class")

        if exclude_sensory:
            return self.internal_neuron_total #return only inter command and motor neuron count
        else:
            return self.internal_neuron_total + self.input_dim #input dim = number of sensory neurons

    #initialise shape of sensory adjacency matrix    
    def _set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros([self.input_dim, self.internal_neuron_total]) #although sensory neurons connect only to inter neurons, we store the entire length of all neurons for consistent matrix sampling as inter neurons are not at the beginning
    
    #set output/ n_classes 
    def _set_output_dim(self, output_dim):
        self.output_dim = output_dim #is equal to the number of motor neurons

    def build(self, input_dim): 
        #error handling of conflicting input size
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(f"Conflicting input dimensions; input dim is: {self.input_dim}, but read {input_dim}")
        
        if self.input_dim is None:
            self._set_input_dim(input_dim)
    
    #creates reverse potential for synapses (sensory and internal) (learnable parameter); input is either neuron_adjacency_matrix or sensory_adjacency_matrix
    def create_synapse_reverse_potential(self, adjacency_matrix, dtype=np.float32):
        return np.copy(adjacency_matrix).astype(dtype) #copy adjacency matrix with npfloat32 datatype
    
    #short cuts for accessing adjacency matrices and synaptic counts
    @property 
    def get_NAM(self):
        return self.neuron_adjacency_matrix 
    
    @property
    def get_SAM(self):
        return self.sensory_adjacency_matrix
    
    @property 
    def synapse_count(self):
        return np.sum(np.abs(self.neuron_adjacency_matrix)) #sum all connections in neuron adjacency
    
    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self.sensory_adjacency_matrix)) #sum all connections in sensory adjacency
    
    #used before adding synapse connection
    def _validate_neuron_connection(self, src, dst, polarity, reference_size, type="neurons"):
        #error handling for src input
        if src < 0 or src >= reference_size: #reference size can be total_neurons for add_synapse or input_dim for add_sensory_synapse
            if type == "neurons":
                print(f"Cannot add neuron connection from neuron {src} when only {reference_size} {type} exist...")
            if type == "features":
                print(f"Cannot add neuron connection from neuron {src} when only {reference_size} {type} exist...")
            return False 
        
        #error handling for dst input
        if dst < 0 or dst >= self.internal_neuron_total:
            print(f"Cannot add connection to destination neuron {dst} when only {self.internal_neuron_total} neurons exist...")
            return False 
        
        if polarity not in [-1, 1]: #must be either -1 or 1
            print(f"Cannot add connection with polarity {polarity} (expected -1 or +1)")
            return False 
        
        return True #input args are valid
    
    #add polarity connectivity for neurons
    def add_synapse_connection(self, src, dst, polarity):
        if self._validate_neuron_connection(src, dst, polarity, self.internal_neuron_total, type="neurons"):
            self.neuron_adjacency_matrix[src, dst] = polarity

    #add for sensory to inter (or other neuron types if NCP class is changed)
    def add_sensory_synapse_connection(self, src, dst, polarity):
        if self._validate_neuron_connection(src, dst, polarity, self.input_dim, type="features"):
            self.sensory_adjacency_matrix[src, dst] = polarity
    
    #storing configuration
    def get_baseline_config(self):
        return {
            'total_neurons': self.internal_neuron_total + self.input_dim,
            'total_internal_neurons': self.internal_neuron_total,
            'total_sensory_neurons': self.input_dim,
            'output_dim': self.output_dim,
            'neuron_adjacency_matrix': self.neuron_adjacency_matrix,
            'sensory_adjacency_matrix': self.sensory_adjacency_matrix
        }
            
#---------------------------
#Neural Circuit Policy (NCP) - creates specific neurons; inter, command and motor; any adjustments to neuron wise architecture gets modified here
#---------------------------  
class NeuralCircuitPolicy(NeuronSynapseWiring):
    def __init__(self, 
                 inter_neurons, #number of interneurons (feature extractors, sensory neurons -> inter neurons)
                 command_neurons, #number of command neurons (recurrent layer; memory/context; inter neurons -> command neurons)
                 motor_neurons, #number of motor neurons (output layer (i.e steering angle, car accel, or new hidden state))
                 outgoing_sensory_neurons, #number of neurons from sensory to inter neurons
                 outgoing_inter_neurons, #number of neurons from inter to command neurons
                 num_of_recurrent_connections, #number of recurrent connections in command neuron layer
                 outgoing_command_neurons, #number of incoming synapses from command to motor neurons
                 seed=24573471): #random seed for producing wiring/connectivity
        
        neuron_total = inter_neurons + command_neurons + motor_neurons #neuron total is 'internal_total_neurons' which excludes sensory (because sensory acts as the 'input layer')

        super(NeuralCircuitPolicy, self).__init__(internal_total_neurons=neuron_total) #initialise neuron adjacency matrix
        self._set_output_dim(motor_neurons) #set output dim = motor neurons; usually 1 or 2
        self.rndm_sd = np.random.RandomState(seed=seed) #reproducable random generator for neuron connectivity

        self._store_internal_neurons(inter_neurons, command_neurons, motor_neurons) #store individual counts into class variables

        self._store_connectivity(outgoing_sensory_neurons, #store connectivity into class variables
                                 outgoing_inter_neurons, 
                                 num_of_recurrent_connections,
                                 outgoing_command_neurons)
        
        self._create_neuron_indicies() #create adjacency matrix indicies for retrieving neurons; order is motor, command, inter
        self._validate_connectivity()

    #using index id, return the type of neuron it is
    def get_neuron_type(self, index):
        if index < self.num_motor_neurons: #because indicies start at motor neurons
            return "motor"
        
        if index < (self.num_motor_neurons + self.num_command_neurons):
            return "command"
        
        if index < (self.num_motor_neurons + self.num_command_neurons + self.num_interneurons):
            return "inter" 
    
        raise ValueError(f"ID {index} is higher than actual number of neuron ids: {(self.num_motor_neurons + self.num_command_neurons + self.num_interneurons) - 1}") #else raise value error that index is too high

    #ensure number of outgoing connections from neurons are within constraints for logical construction
    def _validate_individual_connectivity(self, fanout, constraint, src, dst):
        if fanout > constraint:
            raise ValueError(f"Cannot construct {fanout} outgoing connections from {src} to {dst}, when there are only {constraint} {dst}")

    def _validate_connectivity(self):
        self._validate_individual_connectivity(self.sensory_fanout, self.num_interneurons, src="Sensory Neurons", dst="Interneurons") #sensory_fanout must be <= number of interneurons
        self._validate_individual_connectivity(self.inter_fanout, self.num_command_neurons, src="Interneurons", dst="Command Neurons") #inter fanout must be <= number of command neurons
        self._validate_individual_connectivity(self.command_fanout, self.num_command_neurons, src="Interneurons", dst="Command Neurons") #command fanout must be <= number of command neurons
        
    #create indicies for neuron types in adjacency matrix; motor starts at the beginning because its computationally faster to access its elements during training and inference and only has 1-2 usually
    def _create_neuron_indicies(self):
        motor_indicies = range(self.num_motor_neurons)
        command_indicies = range(self.num_motor_neurons, (self.num_motor_neurons + self.num_command_neurons))
        inter_indicies = range((self.num_motor_neurons + self.num_command_neurons), (self.num_motor_neurons + self.num_command_neurons + self.num_interneurons))

        self.motor_neurons = [i for i in motor_indicies]
        self.command_neurons = [i for i in command_indicies]
        self.interneurons = [i for i in inter_indicies]

    #the number of outgoing connections from each individual neuron type
    def _store_connectivity(self, s_fanout, i_fanout, r_con, c_fanout):
        self.sensory_fanout = s_fanout
        self.inter_fanout = i_fanout
        self.recurrent_connections = r_con 
        self.command_fanout = c_fanout

    #store internal neuron counts
    def _store_internal_neurons(self, inter, com, motor):
        self.num_interneurons = inter 
        self.num_command_neurons = com 
        self.num_motor_neurons = motor

#---------------------------Building connections---------------------------
    def build(self, input_shape): #initialise the synaptic connections between neurons (fill in the adjacency matrices)

        #initialise wiring
        super().build(input_shape) #input_dim 
        self.num_sensory_neurons = self.input_dim #which comes from NeuronWiring
        self.sensory_neurons = [i for i in range(0, self.num_sensory_neurons)] #even tho motor indicies start from 0, sensory neurons have their own adjacency matrix so also start from 0

        #after sensory neurons, build the connections; sensory -> inter -> command -> motor
        self.build_sensory_to_inter_connections()
        self.build_inter_to_command_connections()
        self.build_recurrent_command_layer()
        self.build_command_to_motor_layer()

    #input -> sensory -> inter ->....
    def build_sensory_to_inter_connections(self):
        unreachable_inter_neurons = [neuron_id for neuron_id in self.interneurons] #initialise with all inter neuron ids; ids that are used will be removed from the list dynamically/randomly

        for src in self.sensory_neurons: 
            #selecting random indexes from inter neurons as 'dst' in adjacency matrix, replace=False means it can't randomly select the same index twice, so values are unique
            for dst in self.rndm_sd.choice(self.interneurons, size=self.sensory_fanout, replace=False): 
                if dst in unreachable_inter_neurons:
                    unreachable_inter_neurons.remove(dst) #remove from being unreachable
                
                polarity = self.rndm_sd.choice([-1, 1]) #random excitatory/inhabitory connection
                self.add_sensory_synapse_connection(src, dst, polarity) #add to adjacency matrix
        
        #if some inter neurons are not connected, connect them:
        mean_inter_neuron_fanin = int(self.num_sensory_neurons * self.sensory_fanout / self.num_interneurons) 
        mean_inter_neuron_fanin = np.clip(mean_inter_neuron_fanin, 1, self.num_sensory_neurons) 

        for dest in unreachable_inter_neurons: #looping through remaining neurons
            for src in self.rndm_sd.choice(self.sensory_neurons, size=mean_inter_neuron_fanin, replace=False):
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_sensory_synapse_connection(src, dest, polarity)

    #sensory -> inter -> commands ->.... same process as above, using neuron adjacency matrix
    def build_inter_to_command_connections(self):
        unreachable_command_neurons = [neuron_id for neuron_id in self.command_neurons]

        for src in self.interneurons:
            for dst in self.rndm_sd.choice(self.command_neurons, size=self.inter_fanout, replace=False):
                if dst in unreachable_command_neurons:
                    unreachable_command_neurons.remove(dst)
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_synapse_connection(src, dst, polarity) #add to regular neurons connection adjacency matrix
        
        mean_command_neuron_fain = int(self.num_interneurons * self.inter_fanout / self.num_command_neurons)
        mean_command_neuron_fain = np.clip(mean_command_neuron_fain, 1, self.num_command_neurons)

        for dst in unreachable_command_neurons:
            for src in self.rndm_sd.choice(self.interneurons, size=mean_command_neuron_fain, replace=False):
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_synapse_connection(src, dst, polarity)

    #randomly map connectivity between command neurons for recurrent processing
    def build_recurrent_command_layer(self):
        for i in range(self.recurrent_connections):
            src = self.rndm_sd.choice(self.command_neurons)
            dst = self.rndm_sd.choice(self.command_neurons)
            polarity = self.rndm_sd.choice([-1, 1])
            self.add_synapse_connection(src, dst, polarity)

    #inter -> commands -> motor ->....
    def build_command_to_motor_layer(self):
        unreachable_command_neurons = [neuron_id for neuron_id in self.command_neurons] #using command neuron idxs so that random command neurons connect to motor neurons

        for dst in self.motor_neurons:
            for src in self.rndm_sd.choice(self.command_neurons, size=self.command_fanout, replace=False):
                if src in unreachable_command_neurons:
                    unreachable_command_neurons.remove(src)
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_synapse_connection(src, dst, polarity)
        
        mean_command_fanout = int(self.num_motor_neurons * self.command_fanout / self.num_command_neurons)
        mean_command_fanout = np.clip(mean_command_fanout, 1, self.num_motor_neurons)

        for src in unreachable_command_neurons:
            for dst in self.rndm_sd.choice(self.motor_neurons, size=mean_command_fanout, replace=False):
                polarity = self.rndm_sd.choice([-1, 1])
                self.add_synapse_connection(src, dst, polarity)

    #might need to double check
    def get_config(self):
        return {
            'inter_neurons_ids': self.interneurons,
            'command_neuron_ids': self.command_neurons,
            'motor_neuron_ids': self.motor_neurons,
            'sensory_fanout': self.sensory_fanout,
            'inter_fanout': self.inter_fanout,
            'num_recurrent_connections': self.recurrent_connections,
            'command_fanout': self.command_fanout,
            'seed': self.rndm_sd.seed()
        }