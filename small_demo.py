from small_successful_demo.demo_utilities import load_dataset, load_model, plot_rl_training, decode_stochastic

model_path = r"small_successful_demo\weights\checkpoint_epoch_300.pt"
metrics_path = r"small_successful_demo\metrics"

model = load_model(model_path)
# plot_rl_training(metrics_path)

test = load_dataset(experiments=[11], trials=30, window_size=300) 

for trial_dict in test:
    text_label = trial_dict['texts'][0]
    trial_windows = trial_dict['trial_windows'][0]
    concept_vectors = []
    concept_vector = None #used for previous context argument
    neural_states = None

    for window in trial_windows:
        predictions, certainties_in_thought_steps, final_confidence, neural_states = model(window.unsqueeze(0), neural_states, previous_context=concept_vector)
        concept_vector = predictions[0:, :, -1] #use final prediction as concept vector, its in shape batch x features x thinking steps
        concept_vectors.append(concept_vector)

    #deterministic decoding using argmax for baseline comparison
    sentence_deterministic = model.decode(concept_vectors)
    print(f"Deterministic: {sentence_deterministic}")
    
    #stochastic sampling to reveal vocabulary diversity learned during training
    print("Stochastic samples:")
    for sample_idx in range(5):
        sentence_stochastic = decode_stochastic(model, concept_vectors)
        print(f"  Sample {sample_idx}: {sentence_stochastic}")
    
    print(f"Target: {text_label}")
    print()