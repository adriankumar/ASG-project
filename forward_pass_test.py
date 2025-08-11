import torch
import numpy as np
from data.loading_data import prepare_chisco_dataset, segment_eeg_trial, prepare_windows_for_ctm
from model_components.full_architecture import create_eeg_ctm_model

DEVICE = "cpu"

#configuration for eegnet+ltc feature extraction
FEATURE_EXTRACTOR_CONFIG = {
    'type': 'eegnet-ltc',
    'use_ltc': True,
    'channels': 122,
    'window_size': 200,
    'eegnet_output_dim': 64,
    'ltc': {
        'inter_neurons': 32,
        'command_neurons': 16,
        'motor_neurons': 8,
        'sensory_fanout': 4,
        'inter_fanout': 3,
        'recurrent_connections': 8,
        'command_fanout': 4,
        'input_mapping': 'affine',
        'output_mapping': 'affine',
        'ode_unfolds': 6,
        'implicit_constraints': False
    }
}

#configuration for eegnet-only feature extraction
FEATURE_EXTRACTOR_CONFIG_NO_LTC = {
    'type': 'eegnet',
    'use_ltc': False,
    'channels': 122,
    'window_size': 200,
    'eegnet_output_dim': 128
}

#ctm reasoning configuration
CTM_CONFIG = {
    'num_neurons': 128,
    'memory_length': 24,
    'latent_output_size': 100,
    'latent_action_size': 64,
    'attention_size': 256,
    'num_heads': 8,
    'unet_depth': 6,
    'thinking_steps': 16,
    'output_dim': 768,
    'self_pairing_count': 12,
    'use_deep_nlm': True,
    'use_layernorm': False,
    'dropout': 0.2,
    'temperature': 0.35,
    'min_unet_width': 16
}

#complete model configuration with ltc
CONFIG_WITH_LTC = {
    'feature_extractor': FEATURE_EXTRACTOR_CONFIG,
    'ctm': CTM_CONFIG
}

#complete model configuration without ltc
CONFIG_NO_LTC = {
    'feature_extractor': FEATURE_EXTRACTOR_CONFIG_NO_LTC,
    'ctm': CTM_CONFIG
}

#load single trial for testing
def load_single_trial():
    train_data, _ = prepare_chisco_dataset(
        experiments=[10],
        train_per_exp=1,
        test_per_exp=0,
        channels=122
    )
    
    if len(train_data) == 0:
        raise ValueError("no training data loaded")
    
    eeg_data, text_label = train_data[0]
    print(f"loaded trial with text: '{text_label}'")
    print(f"eeg data shape: {eeg_data.shape}")
    
    return eeg_data, text_label

#convert trial to windowed format for model input
def prepare_trial_windows(eeg_data, window_size=200):
    windows = segment_eeg_trial(eeg_data, window_size)
    torch_windows = prepare_windows_for_ctm(windows)
    
    print(f"segmented into {len(torch_windows)} windows")
    print(f"window shape: {torch_windows[0].shape}")
    
    return torch_windows

#test forward pass through complete model
def test_model_forward(model, trial_windows, text_label):
    model.eval()
    model.to(DEVICE)
    
    print(f"\ntesting sequential processing of {len(trial_windows)} windows...")
    
    #initialise states for sequential processing
    neural_states = None
    previous_context = None
    concept_vectors = []
    
    with torch.no_grad():
        for window_idx, window in enumerate(trial_windows):
            #add batch dimension and move to device
            window_batch = window.unsqueeze(0).to(DEVICE)  #(1, window_size, channels)
            
            print(f"processing window {window_idx + 1}/{len(trial_windows)}")
            print(f"  input shape: {window_batch.shape}")
            
            #forward pass through model
            predictions, certainties, confidence, neural_states = model(
                window_batch, 
                neural_states=neural_states, 
                previous_context=previous_context
            )
            
            #extract final prediction as concept vector
            concept_vector = predictions[0, :, -1]  #(output_dim,)
            concept_vectors.append(concept_vector)
            previous_context = concept_vector.unsqueeze(0)  #add batch dim for next window
            
            print(f"  predictions shape: {predictions.shape}")
            print(f"  certainties shape: {certainties.shape}")
            print(f"  confidence shape: {confidence.shape}")
            print(f"  concept vector shape: {concept_vector.shape}")
            print(f"  confidence score: {confidence[0, 1].item():.4f}")
    
    print(f"\ncollected {len(concept_vectors)} concept vectors")
    return concept_vectors

#test decoding concept vectors to text
def test_decoding(model, concept_vectors):
    print("\ntesting concept vector decoding...")
    
    #deterministic decoding
    gpt2_head, gpt2_tokenizer = model.get_gpt2_components()
    gpt2_head = gpt2_head.to(DEVICE)
    
    tokens = []
    with torch.no_grad():
        for i, vector in enumerate(concept_vectors):
            if torch.all(vector == 0):
                tokens.append('')
            else:
                logits = gpt2_head(vector)
                tok_id = torch.argmax(logits).item()
                token = gpt2_tokenizer.decode([tok_id])
                tokens.append(token)
                print(f"  window {i+1}: token '{token}' (id: {tok_id})")
    
    sentence = ''.join(tokens).replace('', ' ').strip()
    print(f"\nreconstructed sentence: '{sentence}'")
    return sentence

#main testing function
def run_forward_test(use_ltc=True):
    print("=" * 60)
    print(f"testing eeg-ctm forward pass {'with ltc' if use_ltc else 'without ltc'}")
    print("=" * 60)
    
    #select configuration
    config = CONFIG_WITH_LTC if use_ltc else CONFIG_NO_LTC
    
    #create model
    print("creating model...")
    model = create_eeg_ctm_model(config)
    print(f"model created with config: {model.get_config()}")
    
    #load test data
    print("\nloading test data...")
    eeg_data, text_label = load_single_trial()
    
    #prepare windows
    print("\npreparing windowed data...")
    trial_windows = prepare_trial_windows(eeg_data)
    
    #dummy forward pass to initialise lazy layers
    dummy_data = torch.randn(1, 200, 122)
    with torch.no_grad():
        _ = model(dummy_data.to(DEVICE))
    
    #count parameters after initialisation
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    
    #test forward pass
    concept_vectors = test_model_forward(model, trial_windows, text_label)
    
    #test decoding
    reconstructed_text = test_decoding(model, concept_vectors)
    
    #compare with target
    print(f"\ntarget text: '{text_label}'")
    print(f"reconstructed: '{reconstructed_text}'")
    
    print(f"\nforward pass test completed successfully!")
    return model, concept_vectors

def main():
    #test both configurations
    print("testing eegnet+ltc configuration...")
    model_ltc, vectors_ltc = run_forward_test(use_ltc=True)
    
    print("\n" + "=" * 60)
    print("testing eegnet-only configuration...")
    model_no_ltc, vectors_no_ltc = run_forward_test(use_ltc=False)
    
    print("\n" + "=" * 60)
    print("all tests completed successfully!")
    
    #basic comparison
    print(f"ltc model concept vectors: {len(vectors_ltc)}")
    print(f"eegnet model concept vectors: {len(vectors_no_ltc)}")

if __name__ == "__main__":
    main()