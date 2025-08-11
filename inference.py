from utilities.inference_utilities import (
    load_model, load_dataset, inference_pipeline, plot_rl_training_v4
)

DEVICE = "cpu"

#model and metrics paths
model_path = r"rl_training\checkpoint_weights\eegnet_ltc\second_demo\checkpoint_epoch_1500.pt"
metrics_path = r"rl_training\metrics\eegnet_ltc\second_demo"

#load trained eeg-ctm model
model = load_model(model_path, device=DEVICE)
print("eeg-ctm model loaded successfully")

#visualise training metrics including penalties
# print("plotting training metrics...")
# plot_rl_training_v4(metrics_path)

#load test dataset
test_dataset = load_dataset(
    experiments=[10], 
    trials=1, 
    window_size=model.inference_params['window_size'],
    channels=model.inference_params['channels']
)

print(f"\nprocessing {len(test_dataset)} test trials...")

#process each trial with comprehensive inference analysis
for trial_idx, trial_dict in enumerate(test_dataset):
    print(f"\n{'='*60}")
    print(f"trial {trial_idx + 1}/{len(test_dataset)}")
    print(f"{'='*60}")
    
    text_label = trial_dict['texts'][0]
    trial_windows = trial_dict['trial_windows'][0]
    
    #run inference pipeline
    results = inference_pipeline(
        model=model,
        trial_windows=trial_windows,
        target_text=text_label,
        num_stochastic_samples=5,
        device=DEVICE
    )
    
    #display results
    print(f"target sentence: '{text_label}'")
    print(f"deterministic decode: '{results['deterministic']}', with confidence: {results['confidence']}")
    
    print("\nstochastic samples:")
    for i, sample in enumerate(results['stochastic']):
        print(f"  sample {i+1}: '{sample}'")
    
    #analyse vocabulary diversity in stochastic samples
    all_words = set()
    for sample in results['stochastic']:
        all_words.update(sample.lower().split())
    
    print(f"\nvocabulary diversity: {len(all_words)} unique words across samples")
    if len(all_words) > 0:
        print(f"sample vocabulary: {sorted(list(all_words))}")
