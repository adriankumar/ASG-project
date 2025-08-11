import torch
from small_successful_demo.ctm_old import ContinuousThoughtMachine
from small_successful_demo.feature_extractors import EEGFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
from data.loading_data import prepare_chisco_dataset, prepare_torch_dataset
import json
import os

#stochastic decoding using sampling like training for vocabulary diversity analysis
def decode_stochastic(model, sequence_vectors):
    with torch.no_grad():
        gpt2_head, gpt2_tokenizer = model.ctm.get_gpt2_components()
        gpt2_head = gpt2_head.to(model.device)
        
        tokens = []
        for vec in sequence_vectors:
            if torch.all(vec == 0):
                tokens.append('')
            else:
                logits = gpt2_head(vec.to(model.device))
                distribution = torch.distributions.Categorical(logits=logits)
                tok_id = distribution.sample().item()
                token = gpt2_tokenizer.decode([tok_id])
                tokens.append(token)
        
        sentence = ''.join(tokens).strip()
        return sentence

#wrapper that adds decode method
class LoadedCTM:
    def __init__(self, model, feature_extractor):
        self.ctm = model
        self.feature_extractor = feature_extractor
        self.device = torch.device('cpu')

    def __call__(self, eeg_data, neural_states=None, previous_context=None):
        if self.feature_extractor:
            processed = self.feature_extractor(eeg_data).to(self.device)
        else:
            processed = eeg_data.to(self.device)
        return self.ctm(processed, neural_states, previous_context)

    def decode(self, sequence_vectors):
        #decode sequence of concept vectors using ctm's gpt2 components with greedy decoding
        with torch.no_grad():
            gpt2_head, gpt2_tokenizer = self.ctm.get_gpt2_components()
            gpt2_head = gpt2_head.to(self.device)
            
            tokens = []
            for vec in sequence_vectors:
                if torch.all(vec == 0):
                    tokens.append('')
                else:
                    #greedy decoding using argmax for deterministic token selection
                    logits = gpt2_head(vec.to(self.device))
                    tok_id = torch.argmax(logits).item()
                    token = gpt2_tokenizer.decode([tok_id])
                    tokens.append(token)
            
            sentence = ''.join(tokens).strip()
            return sentence

#build and return model ready for use
def load_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    config = checkpoint['ctm_config']
    
    #extract and remove non-ctm parameters before model creation
    feature_extractor_type = config.pop('feature_extractor', 'raw')
    inference_params = {
        'window_size': config.pop('window_size', 200),
        'channels': config.pop('channels', 122),
        'confidence_threshold': config.pop('confidence_threshold', 0.85),
        'fs': config.pop('fs', 250)
    }
    
    #create ctm with aligned configuration keys
    ctm = ContinuousThoughtMachine(**config)
    ctm.load_state_dict(checkpoint['model_state_dict'])
    ctm.eval()

    #create feature extractor based on saved type
    if feature_extractor_type == 'raw':
        extractor = None
    else:
        extractor_kwargs = {'fs': inference_params['fs']} if feature_extractor_type in ['psd', 'fft'] else {}
        extractor = EEGFeatureExtractor(feature_extractor_type, **extractor_kwargs)

    #create wrapper with saved configuration
    wrapper = LoadedCTM(ctm, extractor)
    wrapper.inference_params = inference_params #store for potential use
    return wrapper

#plot policy gradient training progress with enhanced confidence metrics
def plot_rl_training(metrics_dir):
    npz_path = os.path.join(metrics_dir, 'training_metrics.npz')
    meta_path = os.path.join(metrics_dir, 'training_metadata.json')

    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    epochs = data['epochs']
    plt.figure(figsize=(15, 10))
    
    #main metrics subplot
    plt.subplot(2, 3, 1)
    plt.plot(epochs, data['loss'], label='total loss', color='blue')
    plt.plot(epochs, data['reward'], label='reward', color='green')
    plt.plot(epochs, data['confidence'], label='confidence', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('main training metrics')
    plt.legend()
    
    #policy loss subplot
    plt.subplot(2, 3, 2)
    plt.plot(epochs, data['policy_loss'], label='policy loss', color='red')
    plt.xlabel('epoch')
    plt.ylabel('policy loss')
    plt.title('policy gradient loss')
    plt.legend()
    
    #confidence loss subplot
    plt.subplot(2, 3, 3)
    plt.plot(epochs, data['confidence_loss'], label='confidence loss', color='purple')
    plt.xlabel('epoch')
    plt.ylabel('confidence loss')
    plt.title('confidence regularisation loss')
    plt.legend()
    
    #reward focus subplot
    plt.subplot(2, 3, 4)
    plt.plot(epochs, data['reward'], label='semantic similarity reward', color='green', linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.title('reward progression')
    plt.legend()
    
    #confidence statistics subplot
    plt.subplot(2, 3, 5)
    plt.plot(epochs, data['confidence_std'], label='confidence std dev', color='brown')
    plt.xlabel('epoch')
    plt.ylabel('standard deviation')
    plt.title('confidence variability')
    plt.legend()
    
    #windows above threshold subplot
    plt.subplot(2, 3, 6)
    plt.plot(epochs, data['windows_above_threshold'], label='windows above threshold', color='darkblue')
    plt.xlabel('epoch')
    plt.ylabel('percentage (%)')
    plt.title('window usage efficiency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def load_dataset(experiments=[10], trials=1, window_size=200):
    #load and batch dataset for parallel training
    _, test_data = prepare_chisco_dataset(
        experiments=experiments,
        train_per_exp=0,
        test_per_exp=trials
    )
    
    #convert to batched torch dataset
    batched_dataset = prepare_torch_dataset(
        test_data,  
        window_size=window_size,
        batch_size=1
    )
    
    print(f"loaded {len(batched_dataset)} batches for training")
    return batched_dataset