from model_components.ltc.ltccell import LTCCell
from model_components.ltc.neural_wiring import NeuralCircuitPolicy
from model_components.ctm_architecture import ContinuousThoughtMachine
from braindecode.models import EEGNetv4
import torch
import torch.nn as nn

#complete eeg-to-language architecture combining spatial processing, temporal dynamics and reasoning
class EEGCTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.use_ltc = config['feature_extractor']['use_ltc']
        self.channels = config['feature_extractor']['channels']
        self.window_size = config['feature_extractor']['window_size']
        
        self.eegnet = self._init_eegnet()
        self.ltc = self._init_ltc() if self.use_ltc else None
        self.ctm = self._init_ctm()
    
    #initialise eegnet spatial-temporal processor
    def _init_eegnet(self):
        fe_config = self.config['feature_extractor']
        return EEGNetv4(
            n_chans=fe_config['channels'],
            n_outputs=fe_config['eegnet_output_dim'],
            n_times=fe_config['window_size']
        )
    
    #detect spatial feature dimension from eegnet
    def _detect_spatial_dimension(self):
        dummy_input = torch.randn(1, self.window_size, self.channels)
        with torch.no_grad():
            spatial_features = self._extract_spatial_features(dummy_input)
        return spatial_features.shape[-1]
    
    #initialise liquid time-constant cell for temporal dynamics
    def _init_ltc(self):
        ltc_config = self.config['feature_extractor']['ltc']
        
        #auto-detect input dimension from eegnet spatial output
        detected_input_dim = self._detect_spatial_dimension()
        
        #neural circuit policy wiring
        wiring = NeuralCircuitPolicy(
            inter_neurons=ltc_config['inter_neurons'],
            command_neurons=ltc_config['command_neurons'],
            motor_neurons=ltc_config['motor_neurons'],
            outgoing_sensory_neurons=ltc_config['sensory_fanout'],
            outgoing_inter_neurons=ltc_config['inter_fanout'],
            num_of_recurrent_connections=ltc_config['recurrent_connections'],
            outgoing_command_neurons=ltc_config['command_fanout']
        )
        wiring.build(detected_input_dim)
        
        return LTCCell(
            wiring=wiring,
            input_mapping=ltc_config['input_mapping'],
            output_mapping=ltc_config['output_mapping'],
            ode_unfolds=ltc_config['ode_unfolds'],
            implicit_constraints=ltc_config['implicit_constraints']
        )
    
    #initialise continuous thought machine for reasoning
    def _init_ctm(self):
        ctm_config = self.config['ctm']
        return ContinuousThoughtMachine(
            num_neurons=ctm_config['num_neurons'],
            memory_length=ctm_config['memory_length'],
            latent_output_size=ctm_config['latent_output_size'],
            latent_action_size=ctm_config['latent_action_size'],
            attention_size=ctm_config['attention_size'],
            num_heads=ctm_config['num_heads'],
            unet_depth=ctm_config['unet_depth'],
            thinking_steps=ctm_config['thinking_steps'],
            output_dim=ctm_config['output_dim'],
            self_pairing_count=ctm_config['self_pairing_count'],
            use_deep_nlm=ctm_config['use_deep_nlm'],
            use_layernorm=ctm_config['use_layernorm'],
            dropout=ctm_config['dropout'],
            temperature=ctm_config['temperature'],
            min_unet_width=ctm_config['min_unet_width']
        )
    
    #transform input from batch x window_size x channels to eegnet format
    def _transform_input(self, x):
        return x.transpose(1, 2).unsqueeze(1)  #(batch, 1, channels, window_size)
    
    #extract spatial features from eegnet
    def _extract_spatial_features(self, x):
        eeg_input = self._transform_input(x)
        temporal_features = self.eegnet.conv_temporal(eeg_input)
        spatial_features = self.eegnet.conv_spatial(temporal_features)
        return spatial_features.squeeze(2).transpose(1, 2)  #(batch, window_size, features)
    
    #process full eegnet without ltc
    def _process_full_eegnet(self, x):
        # Reuse the same spatial feature extraction (no classification layers)
        spatial_features = self._extract_spatial_features(x)
        
        # Add a projection layer to reach the target dimension (128 vs 64 for LTC)
        # spatial_features shape: (batch, window_size, spatial_feature_dim)
        # We need to project to the configured eegnet_output_dim (128)
        
        if not hasattr(self, 'feature_projection'):
            # Lazy initialization of projection layer
            input_dim = spatial_features.shape[-1]
            target_dim = self.config['feature_extractor']['eegnet_output_dim']  # 128
            self.feature_projection = nn.Linear(input_dim, target_dim).to(spatial_features.device)
        
        projected_features = self.feature_projection(spatial_features)
        return projected_features  # (batch, window_size, 128)
    
    #process spatial features through ltc sequentially
    def _process_ltc_sequential(self, spatial_features, ltc_current_state= None, time_constant=1.0):
        batch_size, seq_len, feature_dim = spatial_features.shape
        
        hidden_states = []
        ltc_state = ltc_current_state
        
        for t in range(seq_len):
            x_t = spatial_features[:, t, :]  #(batch, features)
            outputs, ltc_state = self.ltc(x_t, ltc_state, time_constant)
            hidden_states.append(ltc_state)
        
        return torch.stack(hidden_states, dim=1), ltc_state  #(batch, seq_len, ltc_hidden_dim), and the current ltc state to maintain temporal continuity across segmented windows
    
    #unified feature extraction routing
    def _extract_features(self, x, ltc_current_state=None):
        if self.use_ltc:
            spatial_features = self._extract_spatial_features(x)
            return self._process_ltc_sequential(spatial_features, ltc_current_state)
        else:
            return self._process_full_eegnet(x)
    
    #main forward pass through complete architecture
    def forward(self, x, neural_states=None, previous_context=None, ltc_current_state=None):
        #always return same output structure even if not using ltc feature extractor, then just return none
        if self.use_ltc:
            features, ltc_new_state = self._extract_features(x, ltc_current_state)
            return (*self.ctm(features, neural_states, previous_context), ltc_new_state)
        else:
            features = self._extract_features(x)
            return (*self.ctm(features, neural_states, previous_context), None)
    
    #return configuration for model saving
    def get_config(self):
        config = {
            'feature_extractor': self.config['feature_extractor'],
            'ctm': self.config['ctm']
        }
        
        #add ltc config if used
        if self.use_ltc:
            config['ltc'] = self.config['feature_extractor']['ltc']
        
        return config
    
    #return ctm for checkpoint saving
    def get_ctm(self):
        return self.ctm
    
    #return gpt2 components for decoding
    def get_gpt2_components(self):
        return self.ctm.get_gpt2_components()

#wrapper for training compatibility with existing pipeline
class EEGCTMWrapper(nn.Module):
    def __init__(self, eeg_ctm_model, feature_extractor_type='eegnet-ltc'):
        super().__init__()
        self.eeg_ctm = eeg_ctm_model
        self.feature_extractor_type = feature_extractor_type
    
    def forward(self, eeg_data, neural_states=None, previous_context=None, ltc_current_state=None):
        return self.eeg_ctm(eeg_data, neural_states, previous_context, ltc_current_state)
        
    
    #return complete configuration including feature extraction settings
    def get_config(self):
        config = self.eeg_ctm.get_config()
        config['feature_extractor_type'] = self.feature_extractor_type
        return config
    
    #return ctm parameters for optimiser
    def parameters(self):
        return self.eeg_ctm.parameters()
    
    #move model to device
    def to(self, device):
        self.eeg_ctm.to(device)
        return self
    
    #set training mode
    def train(self):
        self.eeg_ctm.train()
    
    #set evaluation mode
    def eval(self):
        self.eeg_ctm.eval()
    
    def get_gpt2_components(self):
        return self.eeg_ctm.get_gpt2_components()
    
    #access ctm for checkpoint saving
    @property
    def ctm(self):
        return self.eeg_ctm.get_ctm()

#create complete model from configuration
def create_eeg_ctm_model(config):
    eeg_ctm = EEGCTM(config)
    feature_type = 'eegnet-ltc' if config['feature_extractor']['use_ltc'] else 'eegnet'
    return EEGCTMWrapper(eeg_ctm, feature_type)