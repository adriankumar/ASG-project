import torch
from data.loading_data import prepare_chisco_dataset, prepare_torch_dataset
from model_components.full_architecture import create_eeg_ctm_model
from utilities.policy_utilities import train_ctm_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#configuration for eegnet+ltc feature extraction - change use_ltc to False for eegnet-only
FEATURE_EXTRACTOR_CONFIG = {
    'type': 'eegnet-ltc',
    'use_ltc': True,  #set to False for eegnet-only mode; if using ltc, then the input dim for the attention in the ctm is the total number of neurons used for the ltc (inter + command + motor) in shape: batch x window size x ltc_neurons
    'channels': 122,
    'window_size': 300,
    'eegnet_output_dim': 64,  #if using eegnet-only, this output dim becomes the input dim for your attention input dim (uses lazylinear)
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
        'implicit_constraints': True #havent implemented an enforce positive parameter in training so use this
    }
}

#ctm reasoning configuration
CTM_CONFIG = {
    'num_neurons': 128,
    'memory_length': 24,
    'latent_output_size': 100,
    'latent_action_size': 64,
    'attention_size': 256, #attention capacity, not the same as attention input dim
    'num_heads': 8,
    'unet_depth': 6,
    'thinking_steps': 16,
    'output_dim': 768,
    'self_pairing_count': 12,
    'use_deep_nlm': True,
    'use_layernorm': False,
    'dropout': 0.2,
    'temperature': 1.0,
    'min_unet_width': 16
}

DATASET_CONFIG = {
    'window_size': 300,
    'channels': 122,
    'batch_size': 1,
    'confidence_threshold': 0,
    'experiments': [10, 11, 12],
    'train_per_exp': 2,
    'test_per_exp': 0,
    'feature_extractor': 'eegnet-ltc'
}

TRAIN_CONFIG = {
    'epochs': 1500, 
    'model_checkpoint_dir': r"rl_training\checkpoint_weights\eegnet_ltc\second_demo",
    'training_metrics_dir': r"rl_training\metrics\eegnet_ltc\second_demo",
    'loss_weights': {'policy': 0.9, 'confidence': 0.2, 'embedding': 0.1},
    'reward_weights': {
        'semantic': 1.5,  #cosine similarity weight in total reward computation
        'token': 0.0,  #exact word match weight in total reward computation
        'edit': 0.8,  #edit distance weight in total reward computation, structure
        'bleu': 0.5,  #bleu score weight in total reward computation
        'consecutive_penalty': 0.6,  #consecutive penalty weight in total reward computation
        'frequency_penalty': 0.2  #frequency penalty weight in total reward computation
    },
    'penalty_config': {
        'frequency_threshold': 0.3,  #threshold above which words trigger frequency penalty
        'consecutive_penalty_weight': 0.6,  #multiplier for consecutive word penalty strength
        'frequency_penalty_weight': 0.4  #multiplier for frequency penalty strength
    },
    'learning_rate': 0.0005,
    'save_freq': 50
}

#complete model configuration
MODEL_CONFIG = {
    'feature_extractor': FEATURE_EXTRACTOR_CONFIG,
    'ctm': CTM_CONFIG
}

#load and batch dataset for parallel training
def load_batched_dataset():
    train_data, test_data = prepare_chisco_dataset(
        experiments=DATASET_CONFIG['experiments'],
        train_per_exp=DATASET_CONFIG['train_per_exp'],
        test_per_exp=DATASET_CONFIG['test_per_exp'],
        channels=DATASET_CONFIG['channels']
    )
    
    #convert to batched torch dataset
    batched_dataset = prepare_torch_dataset(
        train_data,  
        window_size=DATASET_CONFIG['window_size'],
        batch_size=DATASET_CONFIG['batch_size']
    )
    
    print(f"loaded {len(batched_dataset)} batches for training")
    return batched_dataset

def main():
    #main training execution with complete config persistence
    print("=" * 60)
    print("ctm eeg-to-language training with eegnet+ltc")
    print("=" * 60)
    
    #create wrapped model with feature extraction
    wrapped_model = create_eeg_ctm_model(MODEL_CONFIG)
    
    #load batched dataset
    batched_dataset = load_batched_dataset()
    
    #dummy forward pass to initialise lazy layers
    dummy_data = torch.randn(1, DATASET_CONFIG['window_size'], DATASET_CONFIG['channels'])
    wrapped_model.to(DEVICE)
    with torch.no_grad():
        _ = wrapped_model(dummy_data.to(DEVICE))
    
    #count parameters after initialisation
    total_params = sum(p.numel() for p in wrapped_model.parameters())
    trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    
    #start training with dataset config for checkpoint saving
    print("\nstarting training...")
    metrics_tracker = train_ctm_model(
        model=wrapped_model,
        dataloader=batched_dataset,
        epochs=TRAIN_CONFIG['epochs'],
        model_save_dir=TRAIN_CONFIG['model_checkpoint_dir'],
        metrics_save_dir=TRAIN_CONFIG['training_metrics_dir'],
        loss_weights=TRAIN_CONFIG['loss_weights'],
        reward_weights=TRAIN_CONFIG['reward_weights'],
        learning_rate=TRAIN_CONFIG['learning_rate'],
        save_frequency=TRAIN_CONFIG['save_freq'],
        device=DEVICE,
        dataset_config=DATASET_CONFIG,
        frequency_threshold=TRAIN_CONFIG['penalty_config']['frequency_threshold'],
        consecutive_penalty_weight=TRAIN_CONFIG['penalty_config']['consecutive_penalty_weight'],
        frequency_penalty_weight=TRAIN_CONFIG['penalty_config']['frequency_penalty_weight']
    )
    
    #print final results
    final_metrics = metrics_tracker.get_latest_metrics()
    print("\ntraining completed!")
    print(f"final loss: {final_metrics['loss']:.4f}")
    print(f"final reward: {final_metrics['total_reward']:.3f}")  
    print(f"final confidence: {final_metrics['confidence']:.3f}")
    print(f"final policy loss: {final_metrics['policy_loss']:.4f}")
    print(f"final confidence loss: {final_metrics['confidence_loss']:.4f}")
    print(f"final embedding loss: {final_metrics['embedding_loss']:.4f}")
    print(f"final semantic reward: {final_metrics['semantic_reward']:.3f}")
    print(f"final token reward: {final_metrics['token_reward']:.3f}")
    print(f"final edit reward: {final_metrics['edit_reward']:.3f}")
    print(f"final bleu reward: {final_metrics['bleu_reward']:.3f}")
    print(f"final consecutive penalty: {final_metrics['consecutive_penalty']:.3f}")
    print(f"final frequency penalty: {final_metrics['frequency_penalty']:.3f}")

if __name__ == "__main__":
    main()