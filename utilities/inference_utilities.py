import torch
import numpy as np
import matplotlib.pyplot as plt
from model_components.full_architecture import create_eeg_ctm_model
import json
import os
from data.loading_data import prepare_chisco_dataset, prepare_torch_dataset

#stochastic decoding using sampling like training for vocabulary diversity analysis
def decode_stochastic(model, sequence_vectors, device='cpu'):
    with torch.no_grad():
        gpt2_head, gpt2_tokenizer = model.get_gpt2_components()
        gpt2_head = gpt2_head.to(device)
        
        tokens = []
        for vec in sequence_vectors:
            if torch.all(vec == 0):
                tokens.append('')
            else:
                logits = gpt2_head(vec.to(device))
                distribution = torch.distributions.Categorical(logits=logits)
                tok_id = distribution.sample().item()
                token = gpt2_tokenizer.decode([tok_id])
                tokens.append(token)
        
        sentence = ''.join(tokens).strip()
        return sentence

#deterministic greedy decoding for baseline comparison
def decode_deterministic(model, sequence_vectors, device='cpu'):
    with torch.no_grad():
        gpt2_head, gpt2_tokenizer = model.get_gpt2_components()
        gpt2_head = gpt2_head.to(device)
        
        tokens = []
        for vec in sequence_vectors:
            if torch.all(vec == 0):
                tokens.append('')
            else:
                #greedy decoding using argmax for deterministic token selection
                logits = gpt2_head(vec.to(device))
                tok_id = torch.argmax(logits).item()
                token = gpt2_tokenizer.decode([tok_id])
                tokens.append(token)
        
        sentence = ''.join(tokens).strip()
        return sentence

#process single trial through eeg-ctm model with proper state management
def process_trial_inference(model, trial_windows, device='cpu'):
    model.eval()
    model.to(device)
    
    concept_vectors = []
    neural_states = None
    previous_context = None
    ltc_state = None #for ltc feature extractor 
    
    with torch.no_grad():
        for window in trial_windows:
            #add batch dimension and move to device
            window_batch = window.unsqueeze(0).to(device)
            
            #forward pass through eegnet+ltc+ctm architecture with state persistence
            predictions, certainties, overall_confidence, updated_states, ltc_state = model(
                window_batch, neural_states=neural_states, previous_context=previous_context, ltc_current_state=ltc_state
            )
            
            #extract final prediction as concept vector
            concept_vector = predictions[0, :, -1]  #(output_dim,)
            concept_vectors.append(concept_vector)
            previous_context = concept_vector.unsqueeze(0)  #add batch dim for next window
            
            #update states for next window
            neural_states = updated_states
    
    return concept_vectors, overall_confidence

#inference pipeline with multiple decoding strategies
def inference_pipeline(model, trial_windows, target_text=None, num_stochastic_samples=5, device='cpu'):
    #process trial through model
    concept_vectors, confidence = process_trial_inference(model, trial_windows, device)
    
    #deterministic decoding
    deterministic_result = decode_deterministic(model, concept_vectors, device)
    
    #stochastic sampling
    stochastic_results = []
    for _ in range(num_stochastic_samples):
        stochastic_result = decode_stochastic(model, concept_vectors, device)
        stochastic_results.append(stochastic_result)
    
    results = {
        'deterministic': deterministic_result,
        'stochastic': stochastic_results,
        'target': target_text,
        'concept_vectors': concept_vectors,
        'confidence': confidence
    }
    
    return results

#load complete eeg-ctm model from checkpoint
def load_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config']
    
    #create model from saved configuration
    model = create_eeg_ctm_model(model_config)
    
    #load complete state dict including eegnet, ltc, ctm weights
    model.eeg_ctm.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    #store inference parameters for potential use
    model.inference_params = {
        'window_size': model_config.get('window_size', 200),
        'channels': model_config.get('channels', 122)
    }
    
    print(f"loaded {'eegnet+ltc' if model_config['feature_extractor']['use_ltc'] else 'eegnet-only'} model")
    return model

#load dataset for inference testing
def load_dataset(experiments=[10], trials=1, window_size=200, channels=122):
    _, test_data = prepare_chisco_dataset(
        experiments=experiments,
        train_per_exp=0,
        test_per_exp=trials,
        channels=channels
    )
    
    #convert to batched torch dataset
    batched_dataset = prepare_torch_dataset(
        test_data,  
        window_size=window_size,
        batch_size=1
    )
    
    print(f"loaded {len(batched_dataset)} trials for inference")
    return batched_dataset

#plot training metrics 
#plot comprehensive training metrics including penalties
def plot_rl_training_v4(metrics_dir):
    npz_path = os.path.join(metrics_dir, 'training_metrics.npz')
    meta_path = os.path.join(metrics_dir, 'training_metadata.json')

    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    epochs = data['epochs']
    plt.figure(figsize=(24, 18))
    
    #main metrics subplot
    plt.subplot(4, 4, 1)
    plt.plot(epochs, data['loss'], label='total loss', color='blue', linewidth=2)
    plt.plot(epochs, data['total_reward'], label='total reward', color='green', linewidth=2)
    plt.plot(epochs, data['confidence'], label='confidence', color='orange', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('main training metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #loss components comparison subplot
    plt.subplot(4, 4, 2)
    plt.plot(epochs, data['policy_loss'], label='policy loss', color='red', alpha=0.8)
    plt.plot(epochs, data['confidence_loss'], label='confidence loss', color='purple', alpha=0.8)
    plt.plot(epochs, data['embedding_loss'], label='embedding loss', color='brown', alpha=0.8)
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('loss components breakdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #reward decomposition subplot
    plt.subplot(4, 4, 3)
    plt.plot(epochs, data['semantic_reward'], label='semantic', color='green', linewidth=2)
    plt.plot(epochs, data['token_reward'], label='token match', color='blue', linewidth=2)
    plt.plot(epochs, data['edit_reward'], label='edit distance', color='orange', linewidth=2)
    plt.plot(epochs, data['bleu_reward'], label='bleu score', color='red', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.title('positive reward components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #penalty analysis subplot
    plt.subplot(4, 4, 4)
    plt.plot(epochs, data['consecutive_penalty'], label='consecutive penalty', color='red', linewidth=2)
    plt.plot(epochs, data['frequency_penalty'], label='frequency penalty', color='brown', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('penalty value')
    plt.title('penalty components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #confidence statistics subplot
    plt.subplot(4, 4, 5)
    plt.plot(epochs, data['confidence'], label='mean confidence', color='orange', linewidth=2)
    plt.plot(epochs, data['confidence_std'], label='confidence std dev', color='brown', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('confidence metrics')
    plt.title('confidence behavior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #windows usage efficiency subplot
    plt.subplot(4, 4, 6)
    plt.plot(epochs, data['windows_above_threshold'], label='windows above threshold', color='blue', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('percentage (%)')
    plt.title('window usage efficiency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #weighted reward contributions subplot
    plt.subplot(4, 4, 7)
    semantic_weighted = data['semantic_reward'] * meta['reward_weights']['semantic']
    token_weighted = data['token_reward'] * meta['reward_weights']['token']
    edit_weighted = data['edit_reward'] * meta['reward_weights']['edit']
    bleu_weighted = data['bleu_reward'] * meta['reward_weights']['bleu']
    
    plt.plot(epochs, semantic_weighted, label='semantic weighted', color='green', linewidth=2)
    plt.plot(epochs, token_weighted, label='token weighted', color='blue', alpha=0.8)
    plt.plot(epochs, edit_weighted, label='edit weighted', color='orange', alpha=0.8)
    plt.plot(epochs, bleu_weighted, label='bleu weighted', color='red', alpha=0.8)
    plt.xlabel('epoch')
    plt.ylabel('weighted reward')
    plt.title('weighted reward contributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #penalty impact analysis subplot
    plt.subplot(4, 4, 8)
    consecutive_impact = data['consecutive_penalty'] * meta['reward_weights']['consecutive_penalty']
    frequency_impact = data['frequency_penalty'] * meta['reward_weights']['frequency_penalty']
    
    plt.plot(epochs, consecutive_impact, label='consecutive impact', color='red', linewidth=2)
    plt.plot(epochs, frequency_impact, label='frequency impact', color='brown', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('weighted penalty')
    plt.title('penalty impact on total reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #total reward focus subplot
    plt.subplot(4, 4, 9)
    plt.plot(epochs, data['total_reward'], label='total reward', color='green', linewidth=3)
    #add moving average
    window = min(20, len(epochs)//10)
    if window > 1:
        moving_avg = np.convolve(data['total_reward'], np.ones(window)/window, mode='valid')
        plt.plot(epochs[window-1:], moving_avg, label=f'moving avg ({window})', color='blue', linewidth=2, linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.title('total reward progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #reward stability analysis subplot
    plt.subplot(4, 4, 10)
    reward_variance = np.var([data['semantic_reward'], data['token_reward'], 
                            data['edit_reward'], data['bleu_reward']], axis=0)
    plt.plot(epochs, reward_variance, label='reward variance', color='green', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('variance')
    plt.title('reward component stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #learning progress rate subplot
    plt.subplot(4, 4, 11)
    total_reward_diff = np.diff(data['total_reward'], prepend=data['total_reward'][0])
    plt.plot(epochs, total_reward_diff, label='reward change rate', color='purple', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    #add smoothed trend
    if len(epochs) > 10:
        smooth_diff = np.convolve(total_reward_diff, np.ones(5)/5, mode='same')
        plt.plot(epochs, smooth_diff, label='smoothed trend', color='blue', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('reward change')
    plt.title('learning progress rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #loss vs reward correlation subplot
    plt.subplot(4, 4, 12)
    plt.scatter(data['loss'], data['total_reward'], alpha=0.6, c=epochs, cmap='viridis', s=10)
    plt.colorbar(label='epoch')
    plt.xlabel('total loss')
    plt.ylabel('total reward')
    plt.title('loss vs reward correlation')
    plt.grid(True, alpha=0.3)
    
    #semantic dominance analysis subplot
    plt.subplot(4, 4, 13)
    semantic_contribution = semantic_weighted / (semantic_weighted + token_weighted + edit_weighted + bleu_weighted + 1e-8)
    plt.plot(epochs, semantic_contribution, label='semantic dominance', color='green', linewidth=2)
    plt.axhline(y=meta['reward_weights']['semantic']/sum([meta['reward_weights'][k] for k in ['semantic', 'token', 'edit', 'bleu']]), 
                color='blue', linestyle='--', alpha=0.7, label='expected dominance')
    plt.xlabel('epoch')
    plt.ylabel('proportion')
    plt.title('semantic reward dominance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #penalty effectiveness subplot
    plt.subplot(4, 4, 14)
    total_penalty = consecutive_impact + frequency_impact
    plt.plot(epochs, total_penalty, label='total penalty impact', color='red', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('total penalty')
    plt.title('penalty system effectiveness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #confidence calibration subplot
    plt.subplot(4, 4, 15)
    plt.scatter(data['confidence'], data['total_reward'], alpha=0.6, c=epochs, cmap='plasma', s=10)
    plt.colorbar(label='epoch')
    plt.xlabel('confidence')
    plt.ylabel('total reward')
    plt.title('confidence calibration')
    plt.grid(True, alpha=0.3)
    
    #training efficiency subplot
    plt.subplot(4, 4, 16)
    efficiency = data['total_reward'] / (data['loss'] + 1e-8)
    plt.plot(epochs, efficiency, label='reward/loss ratio', color='purple', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('efficiency')
    plt.title('training efficiency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
