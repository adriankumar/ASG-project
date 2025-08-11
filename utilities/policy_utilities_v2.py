import torch
import numpy as np
import os
import json
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance

#default training configuration
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

#loss component weights for multi-objective training with embedding alignment
DEFAULT_LOSS_WEIGHTS = {
    'policy': 0.5,  #reinforce policy gradient loss for sequence generation
    'confidence': 0.2,  #calibration loss between confidence and actual performance
    'embedding': 0.3  #alignment loss between concept vectors and target token embeddings
}

#reward component weights for multi-objective reward computation with penalties
DEFAULT_REWARD_WEIGHTS = {
    'semantic': 0.3,  #cosine similarity between reconstructed and target sentence embeddings
    'token': 0.25,  #exact word match ratio between reconstructed and target text
    'edit': 0.15,  #normalised edit distance reward for sequence structure quality
    'bleu': 0.1,  #bleu score for phrase-level fluency assessment
    'consecutive_penalty': 0.15,  #penalty for consecutive repeated words to prevent stability-seeking repetition
    'frequency_penalty': 0.05,  #penalty for excessive word frequency to encourage vocabulary diversity
    'substring_penalty': 0.1  #penalty for consecutive substring repetition to prevent concatenation exploits
}

#whitelist words that can repeat without penalty
REPETITION_WHITELIST = {'a', 'an', 'the', 'and', 'or', 'to', 'is', 'are', 'of', 'in', 'on', 'at', 'for', 'with'}

#global sentence encoder for semantic evaluation
sentence_encoder = SentenceTransformer('all-mpnet-base-v2')

class CTMLossManager:
    #handles policy gradient, confidence and embedding alignment loss computation for ctm training with penalties
    def __init__(self, model, loss_weights=None, reward_weights=None, device='cuda', 
                 frequency_threshold=0.4, consecutive_penalty_weight=0.5, frequency_penalty_weight=0.3,
                 substring_penalty_weight=0.4, substring_min_length=4):
        self.model = model #reference to ctm model for gpt2 components access
        self.loss_weights = loss_weights or DEFAULT_LOSS_WEIGHTS
        self.reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS
        self.device = device
        self.baseline = 0.0  #moving average baseline for variance reduction
        self.baseline_momentum = 0.9  #momentum for baseline update
        
        #penalty configuration
        self.frequency_threshold = frequency_threshold #threshold for excessive frequency penalty
        self.consecutive_penalty_weight = consecutive_penalty_weight #weight for consecutive penalties
        self.frequency_penalty_weight = frequency_penalty_weight #weight for frequency penalties
        self.substring_penalty_weight = substring_penalty_weight #weight for substring repetition penalties
        self.substring_min_length = substring_min_length #minimum substring length for repetition detection
    
    #sample tokens from concept vectors and collect log probabilities for policy gradients
    def sample_sequence_with_log_probs(self, concept_vectors):
        gpt2_head, gpt2_tokenizer = self.model.get_gpt2_components()
        gpt2_head = gpt2_head.to(self.device)
        
        sampled_tokens = []
        log_probs = []
        
        for vector in concept_vectors:
            if torch.all(vector == 0):
                sampled_tokens.append('')
                log_probs.append(torch.tensor(0.0, device=self.device, requires_grad=True))
            else:
                logits = gpt2_head(vector)
                distribution = torch.distributions.Categorical(logits=logits)
                sampled_token_id = distribution.sample()
                log_prob = distribution.log_prob(sampled_token_id)
                
                token = gpt2_tokenizer.decode([sampled_token_id.item()])
                sampled_tokens.append(token)
                log_probs.append(log_prob)
        
        reconstructed_text = ''.join(sampled_tokens).strip()
        total_log_prob = torch.stack(log_probs).sum()
        
        return reconstructed_text, total_log_prob
    
    #deterministic argmax decoding for penalty computation and embedding alignment
    def _decode_deterministic_text(self, concept_vectors):
        gpt2_head, gpt2_tokenizer = self.model.get_gpt2_components()
        gpt2_head = gpt2_head.to(self.device)
        
        decoded_tokens = []
        for vector in concept_vectors:
            if torch.all(vector == 0):
                decoded_tokens.append('')
            else:
                logits = gpt2_head(vector)
                predicted_token_id = torch.argmax(logits).item()
                token = gpt2_tokenizer.decode([predicted_token_id])
                decoded_tokens.append(token)
        
        reconstructed_text = ''.join(decoded_tokens).strip()
        return reconstructed_text
    
    #decode concept vectors with optional sampling for training or deterministic for inference
    def decode_concept_vectors(self, concept_vectors, sample_for_training=False):
        if sample_for_training:
            return self.sample_sequence_with_log_probs(concept_vectors)
        
        #deterministic decoding for inference/evaluation
        gpt2_head, gpt2_tokenizer = self.model.get_gpt2_components()
        gpt2_head = gpt2_head.to(self.device)
        
        decoded_tokens = []
        for vector in concept_vectors:
            if torch.all(vector == 0):
                decoded_tokens.append('')
            else:
                logits = gpt2_head(vector)
                predicted_token_id = torch.argmax(logits).item()
                token = gpt2_tokenizer.decode([predicted_token_id])
                decoded_tokens.append(token)
        
        reconstructed_sentence = ''.join(decoded_tokens).strip()
        return decoded_tokens, reconstructed_sentence
    
    #compute cosine similarity between reconstructed and target sentences using sampling
    def compute_semantic_similarity(self, reconstructed_text, target_text):
        global sentence_encoder
        reconstructed_embedding = sentence_encoder.encode([reconstructed_text])
        target_embedding = sentence_encoder.encode([target_text])
        
        similarity = cosine_similarity(reconstructed_embedding, target_embedding)[0][0]
        return similarity
    
    #compute exact word match ratio between reconstructed and target text using sampling
    def compute_token_exact_match(self, reconstructed_text, target_text):
        reconstructed_words = reconstructed_text.lower().split()
        target_words = target_text.lower().split()
        
        if len(target_words) == 0:
            return 0.0
        
        matches = sum(1 for word in reconstructed_words if word in target_words)
        return matches / len(target_words)
    
    #compute edit distance reward with normalisation using sampling
    def compute_edit_distance_reward(self, reconstructed_text, target_text):
        reconstructed_words = reconstructed_text.lower().split()
        target_words = target_text.lower().split()
        
        if len(target_words) == 0:
            return 1.0 if len(reconstructed_words) == 0 else 0.0
        
        distance = edit_distance(reconstructed_words, target_words)
        max_distance = max(len(reconstructed_words), len(target_words))
        
        #convert distance to reward normalised between 0 and 1
        reward = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        return reward
    
    #compute bleu score between reconstructed and target text using sampling
    def compute_bleu_score(self, reconstructed_text, target_text):
        reconstructed_words = reconstructed_text.lower().split()
        target_words = target_text.lower().split()
        
        if len(target_words) == 0:
            return 1.0 if len(reconstructed_words) == 0 else 0.0
        
        #bleu score with smoothing for short sequences
        bleu = sentence_bleu([target_words], reconstructed_words, 
                           weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
        return bleu
    
    #compute penalty for consecutive repeated words using deterministic decoding to target inference behavior
    def compute_consecutive_penalty(self, concept_vectors):
        deterministic_text = self._decode_deterministic_text(concept_vectors)
        words = deterministic_text.lower().split()
        if len(words) <= 1:
            return 0.0
        
        consecutive_count = 0
        for i in range(len(words) - 1):
            word = words[i]
            next_word = words[i + 1]
            
            #skip penalty if word is in whitelist
            if word in REPETITION_WHITELIST:
                continue
                
            if word == next_word:
                consecutive_count += 1
        
        #normalise by sentence length and apply weight
        penalty = (consecutive_count / len(words)) * self.consecutive_penalty_weight
        return penalty
    
    #compute penalty for excessive word frequency using deterministic decoding to target inference behavior
    def compute_frequency_penalty(self, concept_vectors):
        deterministic_text = self._decode_deterministic_text(concept_vectors)
        words = deterministic_text.lower().split()
        if len(words) == 0:
            return 0.0
        
        #count word frequencies
        word_counts = {}
        for word in words:
            if word not in REPETITION_WHITELIST:  #skip whitelisted words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        #compute penalty for words above threshold
        total_penalty = 0.0
        for word, count in word_counts.items():
            frequency = count / len(words)
            if frequency > self.frequency_threshold:
                excess_frequency = frequency - self.frequency_threshold
                total_penalty += excess_frequency * self.frequency_penalty_weight
        
        return total_penalty
    
    #detect consecutive substring repetitions to prevent concatenation exploits
    def _find_consecutive_substring_repetitions(self, text):
        if len(text) < self.substring_min_length * 2:
            return 0
        
        max_repetitions = 0
        text_lower = text.lower()
        
        #check all possible substring positions and lengths
        for start_pos in range(len(text_lower) - self.substring_min_length + 1):
            for substr_len in range(self.substring_min_length, len(text_lower) - start_pos + 1):
                substr = text_lower[start_pos:start_pos + substr_len]
                
                #count consecutive repetitions of this substring
                current_pos = start_pos
                repetitions = 0
                
                while current_pos + substr_len <= len(text_lower):
                    if text_lower[current_pos:current_pos + substr_len] == substr:
                        repetitions += 1
                        current_pos += substr_len
                    else:
                        break
                
                #update maximum repetitions found
                if repetitions > max_repetitions:
                    max_repetitions = repetitions
        
        return max_repetitions
    
    #compute penalty for consecutive substring repetitions using deterministic decoding to target inference behavior
    def compute_substring_repetition_penalty(self, concept_vectors):
        deterministic_text = self._decode_deterministic_text(concept_vectors)
        if len(deterministic_text) < self.substring_min_length * 2:
            return 0.0
        
        max_repetitions = self._find_consecutive_substring_repetitions(deterministic_text)
        
        #apply exponential penalty scaling for multiple repetitions
        if max_repetitions <= 1:
            return 0.0
        
        #exponential scaling: penalty grows rapidly with more repetitions
        penalty = self.substring_penalty_weight * (2.0 ** (max_repetitions - 2))
        return penalty
    
    #compute combined reward from multiple text-based metrics including deterministic penalties
    def compute_total_reward(self, sampled_text, target_text, concept_vectors):
        #sampling-based rewards for exploration
        semantic_reward = self.compute_semantic_similarity(sampled_text, target_text)
        token_reward = self.compute_token_exact_match(sampled_text, target_text)
        edit_reward = self.compute_edit_distance_reward(sampled_text, target_text)
        bleu_reward = self.compute_bleu_score(sampled_text, target_text)
        
        #deterministic penalties targeting inference behavior
        consecutive_penalty = self.compute_consecutive_penalty(concept_vectors)
        frequency_penalty = self.compute_frequency_penalty(concept_vectors)
        substring_penalty = self.compute_substring_repetition_penalty(concept_vectors)
        
        #compute base reward
        base_reward = (self.reward_weights['semantic'] * semantic_reward +
                      self.reward_weights['token'] * token_reward +
                      self.reward_weights['edit'] * edit_reward +
                      self.reward_weights['bleu'] * bleu_reward)
        
        #apply penalties multiplicatively
        penalty_factor = (self.reward_weights['consecutive_penalty'] * consecutive_penalty +
                         self.reward_weights['frequency_penalty'] * frequency_penalty +
                         self.reward_weights['substring_penalty'] * substring_penalty)
        
        total_reward = base_reward * (1.0 - penalty_factor)
        
        return total_reward, semantic_reward, token_reward, edit_reward, bleu_reward, consecutive_penalty, frequency_penalty, substring_penalty
    
    #compute target token embeddings from ground truth text
    def compute_target_embeddings(self, target_text):
        gpt2_model = self.model.ctm.get_gpt2_model()
        _, gpt2_tokenizer = self.model.get_gpt2_components()
        
        target_tokens = gpt2_tokenizer.encode(target_text)
        token_tensor = torch.tensor(target_tokens, device=self.device)
        target_embeddings = gpt2_model.transformer.wte(token_tensor)
        
        return target_embeddings
    
    #compute cosine similarity loss between concept vectors and target token embeddings using deterministic decoding
    def compute_embedding_alignment_loss(self, concept_vectors, target_text):
        target_embeddings = self.compute_target_embeddings(target_text)
        
        #handle sequence length mismatch using minimum length
        min_len = min(len(concept_vectors), len(target_embeddings))
        if min_len == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        concept_stack = torch.stack(concept_vectors[:min_len])
        target_stack = target_embeddings[:min_len]
        
        embedding_loss = 1 - F.cosine_similarity(concept_stack, target_stack, dim=-1).sum()
        return embedding_loss
    
    #compute reinforce policy gradient loss using multi-objective reward with deterministic penalties
    def compute_policy_loss(self, concept_vectors, target_text):
        sampled_text, sequence_log_prob = self.sample_sequence_with_log_probs(concept_vectors)
        total_reward, semantic_reward, token_reward, edit_reward, bleu_reward, consecutive_penalty, frequency_penalty, substring_penalty = self.compute_total_reward(sampled_text, target_text, concept_vectors)
        
        #update baseline using exponential moving average
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * total_reward
        
        #reinforce gradient with baseline for variance reduction
        advantage = total_reward - self.baseline
        policy_loss = -sequence_log_prob * advantage
        
        return policy_loss, total_reward, semantic_reward, token_reward, edit_reward, bleu_reward, consecutive_penalty, frequency_penalty, substring_penalty
    
    #calibration loss to encourage confidence matching actual performance
    def compute_confidence_loss(self, overall_confidence, total_reward):
        confidence_score = overall_confidence[1]
        #squared error between confidence and total reward for proper calibration
        confidence_loss = (confidence_score - total_reward) ** 2
        return confidence_loss
    
    #compute combined policy gradient, confidence and embedding alignment loss for single trial
    def compute_trial_loss(self, concept_vectors, target_text, overall_confidence):
        policy_loss, total_reward, semantic_reward, token_reward, edit_reward, bleu_reward, consecutive_penalty, frequency_penalty, substring_penalty = self.compute_policy_loss(concept_vectors, target_text)
        confidence_loss = self.compute_confidence_loss(overall_confidence, total_reward)
        embedding_loss = self.compute_embedding_alignment_loss(concept_vectors, target_text)
        
        trial_loss = (self.loss_weights['policy'] * policy_loss + 
                     self.loss_weights['confidence'] * confidence_loss +
                     self.loss_weights['embedding'] * embedding_loss)
        
        return trial_loss, policy_loss.item(), confidence_loss.item(), embedding_loss.item(), total_reward, semantic_reward, token_reward, edit_reward, bleu_reward, consecutive_penalty, frequency_penalty, substring_penalty
    
    #aggregate trial losses for batch gradient computation
    def compute_batch_loss(self, trial_losses):
        batch_loss = torch.stack(trial_losses).mean()
        return batch_loss
    
class CTMMetricsTracker:
    #tracks training metrics across epochs including embedding alignment loss and multi-objective rewards with penalties
    def __init__(self):
        self.loss_history = []  #combined training loss from policy, confidence and embedding components
        self.confidence_history = []  #average confidence scores across windows
        self.total_reward_history = []  #combined reward including all components and penalties
        self.semantic_reward_history = []  #cosine similarity between sentence embeddings
        self.token_reward_history = []  #exact word match ratio for vocabulary precision
        self.edit_reward_history = []  #edit distance reward for sequence structure quality
        self.bleu_reward_history = []  #bleu score for phrase-level fluency
        self.consecutive_penalty_history = []  #penalty for consecutive repeated words
        self.frequency_penalty_history = []  #penalty for excessive word frequency
        self.substring_penalty_history = []  #penalty for consecutive substring repetitions
        self.policy_loss_history = []  #reinforce policy gradient loss for sequence generation
        self.confidence_loss_history = []  #calibration loss between confidence and performance
        self.embedding_loss_history = []  #alignment loss between concept vectors and token embeddings
        self.confidence_std_history = []  #standard deviation of confidence scores for stability tracking
        self.windows_above_threshold_history = []  #percentage of windows above confidence threshold for usage efficiency
        
    def update_epoch_metrics(self, avg_loss, avg_confidence, avg_total_reward, avg_semantic_reward,
                           avg_token_reward, avg_edit_reward, avg_bleu_reward, avg_consecutive_penalty,
                           avg_frequency_penalty, avg_substring_penalty, avg_policy_loss, avg_confidence_loss, 
                           avg_embedding_loss, avg_confidence_std, avg_windows_above_threshold):
        #store epoch-level aggregated metrics including penalties
        self.loss_history.append(avg_loss)
        self.confidence_history.append(avg_confidence)
        self.total_reward_history.append(avg_total_reward)
        self.semantic_reward_history.append(avg_semantic_reward)
        self.token_reward_history.append(avg_token_reward)
        self.edit_reward_history.append(avg_edit_reward)
        self.bleu_reward_history.append(avg_bleu_reward)
        self.consecutive_penalty_history.append(avg_consecutive_penalty)
        self.frequency_penalty_history.append(avg_frequency_penalty)
        self.substring_penalty_history.append(avg_substring_penalty)
        self.policy_loss_history.append(avg_policy_loss)
        self.confidence_loss_history.append(avg_confidence_loss)
        self.embedding_loss_history.append(avg_embedding_loss)
        self.confidence_std_history.append(avg_confidence_std)
        self.windows_above_threshold_history.append(avg_windows_above_threshold)
    
    def get_latest_metrics(self):
        #return most recent epoch metrics including penalties
        return {
            'loss': self.loss_history[-1],
            'confidence': self.confidence_history[-1],
            'total_reward': self.total_reward_history[-1],
            'semantic_reward': self.semantic_reward_history[-1],
            'token_reward': self.token_reward_history[-1],
            'edit_reward': self.edit_reward_history[-1],
            'bleu_reward': self.bleu_reward_history[-1],
            'consecutive_penalty': self.consecutive_penalty_history[-1],
            'frequency_penalty': self.frequency_penalty_history[-1],
            'substring_penalty': self.substring_penalty_history[-1],
            'policy_loss': self.policy_loss_history[-1],
            'confidence_loss': self.confidence_loss_history[-1],
            'embedding_loss': self.embedding_loss_history[-1],
            'confidence_std': self.confidence_std_history[-1],
            'windows_above_threshold': self.windows_above_threshold_history[-1]
        }

def process_batch_parallel(model, batch_dict, loss_manager, 
                          confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    #process batch of trials in parallel and compute batch loss using multi-objective training with penalties
    device = next(model.parameters()).device
    batch_trial_windows = batch_dict['trial_windows']
    batch_texts = batch_dict['texts']
    
    batch_size = len(batch_trial_windows)
    num_windows = len(batch_trial_windows[0])
    
    #initialise neural states for all trials in batch
    neural_states = None
    
    #accumulate concept vectors for each trial across windows
    batch_concept_vectors = [[] for _ in range(batch_size)]
    batch_confidences = [[] for _ in range(batch_size)]
    all_confidences = []  #track all confidence values for statistics
    windows_above_threshold = 0  #count windows above threshold
    total_windows = 0  #total window count
    
    #process each window position in parallel across all trials
    previous_context = None #for contextual reasoning
    ltc_state = None #for ltc feature extractor continuous temporal dynamics

    for window_position in range(num_windows):
        #stack windows from all trials at current position
        window_batch = torch.stack([
            batch_trial_windows[trial_idx][window_position]
            for trial_idx in range(batch_size)
        ]).to(device)
        
        #forward pass through eegnet+ltc+ctm architecture with state persistence
        predictions, certainties, overall_confidence, updated_states, ltc_state = model(
                window_batch, neural_states=neural_states, previous_context=previous_context, ltc_current_state=ltc_state
            )

        
        #extract final predictions and confidence scores
        final_predictions = predictions[:, :, -1]  #shape: (batch_size, output_dim)
        previous_context = final_predictions.detach()  #detach to prevent gradient chain issues
        confidence_scores = overall_confidence[:, 1]  #shape: (batch_size,)
        
        #apply confidence gating and store concept vectors per trial
        for trial_idx in range(batch_size):
            trial_prediction = final_predictions[trial_idx]
            trial_confidence = confidence_scores[trial_idx].item()
            
            #collect confidence statistics
            all_confidences.append(trial_confidence)
            total_windows += 1
            if trial_confidence > confidence_threshold:
                windows_above_threshold += 1
                batch_concept_vectors[trial_idx].append(trial_prediction)
            else:
                blank_vector = torch.zeros_like(trial_prediction)
                batch_concept_vectors[trial_idx].append(blank_vector)
            
            batch_confidences[trial_idx].append(trial_confidence)
        
        #update neural states for next window position
        neural_states = updated_states
    
    #compute confidence statistics
    confidence_std = np.std(all_confidences)
    windows_above_threshold_pct = (windows_above_threshold / total_windows) * 100
    
    #compute losses for each trial in batch using multi-objective training with penalties
    trial_losses = []
    batch_policy_losses = []
    batch_confidence_losses = []
    batch_embedding_losses = []
    batch_avg_confidences = []
    batch_total_rewards = []
    batch_semantic_rewards = []
    batch_token_rewards = []
    batch_edit_rewards = []
    batch_bleu_rewards = []
    batch_consecutive_penalties = []
    batch_frequency_penalties = []
    batch_substring_penalties = []
    
    for trial_idx in range(batch_size):
        trial_concept_vectors = batch_concept_vectors[trial_idx]
        trial_text = batch_texts[trial_idx]
        trial_overall_confidence = overall_confidence[trial_idx]
        
        #compute trial loss using multi-objective approach with penalties
        trial_loss, policy_loss, confidence_loss, embedding_loss, total_reward, semantic_reward, token_reward, edit_reward, bleu_reward, consecutive_penalty, frequency_penalty, substring_penalty = loss_manager.compute_trial_loss(
            trial_concept_vectors, trial_text, trial_overall_confidence
        )
        
        trial_losses.append(trial_loss)
        batch_policy_losses.append(policy_loss)
        batch_confidence_losses.append(confidence_loss)
        batch_embedding_losses.append(embedding_loss)
        batch_avg_confidences.append(np.mean(batch_confidences[trial_idx]))
        batch_total_rewards.append(total_reward)
        batch_semantic_rewards.append(semantic_reward)
        batch_token_rewards.append(token_reward)
        batch_edit_rewards.append(edit_reward)
        batch_bleu_rewards.append(bleu_reward)
        batch_consecutive_penalties.append(consecutive_penalty)
        batch_frequency_penalties.append(frequency_penalty)
        batch_substring_penalties.append(substring_penalty)
    
    #return batch metrics including penalties
    return {
        'trial_losses': trial_losses,
        'policy_losses': batch_policy_losses,
        'confidence_losses': batch_confidence_losses,
        'embedding_losses': batch_embedding_losses,
        'confidences': batch_avg_confidences,
        'total_rewards': batch_total_rewards,
        'semantic_rewards': batch_semantic_rewards,
        'token_rewards': batch_token_rewards,
        'edit_rewards': batch_edit_rewards,
        'bleu_rewards': batch_bleu_rewards,
        'consecutive_penalties': batch_consecutive_penalties,
        'frequency_penalties': batch_frequency_penalties,
        'substring_penalties': batch_substring_penalties,
        'confidence_std': confidence_std,
        'windows_above_threshold_pct': windows_above_threshold_pct
    }

def process_training_batch(model, batch_dict, loss_manager, optimizer, 
                          gradient_clip=DEFAULT_GRADIENT_CLIP, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    #process batch of trials with parallel window processing and gradient update
    batch_metrics = process_batch_parallel(model, batch_dict, loss_manager, confidence_threshold=confidence_threshold)
    
    #compute batch loss and perform gradient update
    batch_loss = loss_manager.compute_batch_loss(batch_metrics['trial_losses'])
    
    optimizer.zero_grad()
    batch_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    
    #return aggregated batch metrics including penalties
    return {
        'loss': batch_loss.item(),
        'policy_loss': np.mean(batch_metrics['policy_losses']),
        'confidence_loss': np.mean(batch_metrics['confidence_losses']),
        'embedding_loss': np.mean(batch_metrics['embedding_losses']),
        'confidence': np.mean(batch_metrics['confidences']),
        'total_reward': np.mean(batch_metrics['total_rewards']),
        'semantic_reward': np.mean(batch_metrics['semantic_rewards']),
        'token_reward': np.mean(batch_metrics['token_rewards']),
        'edit_reward': np.mean(batch_metrics['edit_rewards']),
        'bleu_reward': np.mean(batch_metrics['bleu_rewards']),
        'consecutive_penalty': np.mean(batch_metrics['consecutive_penalties']),
        'frequency_penalty': np.mean(batch_metrics['frequency_penalties']),
        'substring_penalty': np.mean(batch_metrics['substring_penalties']),
        'confidence_std': batch_metrics['confidence_std'],
        'windows_above_threshold_pct': batch_metrics['windows_above_threshold_pct']
    }

def train_epoch(model, dataloader, loss_manager, optimizer, metrics_tracker, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    #train single epoch with parallel batch processing using multi-objective training with penalties
    model.train()
    
    epoch_losses = []
    epoch_policy_losses = []
    epoch_confidence_losses = []
    epoch_embedding_losses = []
    epoch_confidences = []
    epoch_total_rewards = []
    epoch_semantic_rewards = []
    epoch_token_rewards = []
    epoch_edit_rewards = []
    epoch_bleu_rewards = []
    epoch_consecutive_penalties = []
    epoch_frequency_penalties = []
    epoch_substring_penalties = []
    epoch_confidence_stds = []
    epoch_windows_above_threshold = []
    
    #process all batches in epoch
    for batch_dict in dataloader:
        batch_metrics = process_training_batch(model, batch_dict, loss_manager, optimizer, confidence_threshold=confidence_threshold)
        
        epoch_losses.append(batch_metrics['loss'])
        epoch_policy_losses.append(batch_metrics['policy_loss'])
        epoch_confidence_losses.append(batch_metrics['confidence_loss'])
        epoch_embedding_losses.append(batch_metrics['embedding_loss'])
        epoch_confidences.append(batch_metrics['confidence'])
        epoch_total_rewards.append(batch_metrics['total_reward'])
        epoch_semantic_rewards.append(batch_metrics['semantic_reward'])
        epoch_token_rewards.append(batch_metrics['token_reward'])
        epoch_edit_rewards.append(batch_metrics['edit_reward'])
        epoch_bleu_rewards.append(batch_metrics['bleu_reward'])
        epoch_consecutive_penalties.append(batch_metrics['consecutive_penalty'])
        epoch_frequency_penalties.append(batch_metrics['frequency_penalty'])
        epoch_substring_penalties.append(batch_metrics['substring_penalty'])
        epoch_confidence_stds.append(batch_metrics['confidence_std'])
        epoch_windows_above_threshold.append(batch_metrics['windows_above_threshold_pct'])
    
    #compute epoch averages including penalties
    avg_loss = np.mean(epoch_losses)
    avg_policy_loss = np.mean(epoch_policy_losses)
    avg_confidence_loss = np.mean(epoch_confidence_losses)
    avg_embedding_loss = np.mean(epoch_embedding_losses)
    avg_confidence = np.mean(epoch_confidences)
    avg_total_reward = np.mean(epoch_total_rewards)
    avg_semantic_reward = np.mean(epoch_semantic_rewards)
    avg_token_reward = np.mean(epoch_token_rewards)
    avg_edit_reward = np.mean(epoch_edit_rewards)
    avg_bleu_reward = np.mean(epoch_bleu_rewards)
    avg_consecutive_penalty = np.mean(epoch_consecutive_penalties)
    avg_frequency_penalty = np.mean(epoch_frequency_penalties)
    avg_substring_penalty = np.mean(epoch_substring_penalties)
    avg_confidence_std = np.mean(epoch_confidence_stds)
    avg_windows_above_threshold = np.mean(epoch_windows_above_threshold)
    
    #update metrics tracker including penalties
    metrics_tracker.update_epoch_metrics(
        avg_loss, avg_confidence, avg_total_reward, avg_semantic_reward, avg_token_reward,
        avg_edit_reward, avg_bleu_reward, avg_consecutive_penalty, avg_frequency_penalty, avg_substring_penalty,
        avg_policy_loss, avg_confidence_loss, avg_embedding_loss, avg_confidence_std, avg_windows_above_threshold
    )
    
    return {
        'loss': avg_loss,
        'policy_loss': avg_policy_loss,
        'confidence_loss': avg_confidence_loss,
        'embedding_loss': avg_embedding_loss,
        'confidence': avg_confidence,
        'total_reward': avg_total_reward,
        'semantic_reward': avg_semantic_reward,
        'token_reward': avg_token_reward,
        'edit_reward': avg_edit_reward,
        'bleu_reward': avg_bleu_reward,
        'consecutive_penalty': avg_consecutive_penalty,
        'frequency_penalty': avg_frequency_penalty,
        'substring_penalty': avg_substring_penalty,
        'confidence_std': avg_confidence_std,
        'windows_above_threshold': avg_windows_above_threshold
    }

def train_ctm_model(model, dataloader, epochs, model_save_dir, metrics_save_dir, 
                   loss_weights=None, reward_weights=None, learning_rate=DEFAULT_LEARNING_RATE,
                   save_frequency=10, device='cuda', dataset_config=None, 
                   frequency_threshold=0.4, consecutive_penalty_weight=0.5, frequency_penalty_weight=0.3,
                   substring_penalty_weight=0.4, substring_min_length=4):
    #main training loop for ctm model with multi-objective training and penalties
    
    loss_manager = CTMLossManager(
        model, loss_weights, reward_weights, device,
        frequency_threshold=frequency_threshold,
        consecutive_penalty_weight=consecutive_penalty_weight,
        frequency_penalty_weight=frequency_penalty_weight,
        substring_penalty_weight=substring_penalty_weight,
        substring_min_length=substring_min_length
    )
    metrics_tracker = CTMMetricsTracker()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #ensure save directory exists
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)

    print(f"starting ctm multi-objective training with penalties for {epochs} epochs")
    
    #main epoch loop
    epoch_bar = tqdm(range(epochs), desc="training epochs")

    initial_conf_threshold = dataset_config['confidence_threshold']
    
    for epoch in epoch_bar:
        #dynamic confidence thresholding - commented for later testing
        # conf_threshold = initial_conf_threshold + (0.85 - initial_conf_threshold) * (epoch / epochs)
        conf_threshold = initial_conf_threshold

        epoch_metrics = train_epoch(model, dataloader, loss_manager, optimizer, metrics_tracker, confidence_threshold=conf_threshold)
        
        epoch_bar.set_postfix(
            loss=f"{epoch_metrics['loss']:.4f}",
            reward=f"{epoch_metrics['total_reward']:.3f}",
            cons_pen=f"{epoch_metrics['consecutive_penalty']:.3f}",
            freq_pen=f"{epoch_metrics['frequency_penalty']:.3f}",
            subs_pen=f"{epoch_metrics['substring_penalty']:.3f}",
            conf=f"{epoch_metrics['confidence']:.3f}"
        )
        
        #save checkpoint with dataset config for complete reproducibility
        if (epoch + 1) % save_frequency == 0:
            save_ctm_checkpoint(model, model_save_dir, epoch + 1, dataset_config)
            #save metrics history
            save_metrics_history(metrics_tracker, metrics_save_dir, loss_weights, reward_weights)

    save_metrics_history(metrics_tracker, metrics_save_dir, loss_weights, reward_weights)

    print("training completed")
    return metrics_tracker

def save_ctm_checkpoint(model, save_dir, epoch, dataset_config=None):
    #save complete model checkpoint with architecture configuration for reconstruction
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    
    #extract complete configuration from wrapped model
    complete_config = model.get_config()
    
    #add only essential inference parameters for input compatibility
    if dataset_config:
        inference_params = {
            'window_size': dataset_config.get('window_size', 200),
            'channels': dataset_config.get('channels', 122)
        }
        complete_config.update(inference_params)
    
    checkpoint = {
        'model_state_dict': model.eeg_ctm.state_dict(), #save complete eegctm state including eegnet, ltc, ctm
        'model_config': complete_config,
    }
    
    torch.save(checkpoint, checkpoint_path)

def save_metrics_history(metrics_tracker, save_dir, loss_weights=None, reward_weights=None):
    #save training metrics history including penalties
    metrics_path = os.path.join(save_dir, 'training_metrics.npz')
    
    np.savez(metrics_path,
             epochs=np.arange(1, len(metrics_tracker.loss_history) + 1),
             loss=np.array(metrics_tracker.loss_history),
             confidence=np.array(metrics_tracker.confidence_history),
             total_reward=np.array(metrics_tracker.total_reward_history),
             semantic_reward=np.array(metrics_tracker.semantic_reward_history),
             token_reward=np.array(metrics_tracker.token_reward_history),
             edit_reward=np.array(metrics_tracker.edit_reward_history),
             bleu_reward=np.array(metrics_tracker.bleu_reward_history),
             consecutive_penalty=np.array(metrics_tracker.consecutive_penalty_history),
             frequency_penalty=np.array(metrics_tracker.frequency_penalty_history),
             substring_penalty=np.array(metrics_tracker.substring_penalty_history),
             policy_loss=np.array(metrics_tracker.policy_loss_history),
             confidence_loss=np.array(metrics_tracker.confidence_loss_history),
             embedding_loss=np.array(metrics_tracker.embedding_loss_history),
             confidence_std=np.array(metrics_tracker.confidence_std_history),
             windows_above_threshold=np.array(metrics_tracker.windows_above_threshold_history))
    
    #save metadata including penalties
    metadata = {
        'num_epochs': len(metrics_tracker.loss_history),
        'loss_weights': loss_weights if loss_weights is not None else DEFAULT_LOSS_WEIGHTS,
        'reward_weights': reward_weights if reward_weights is not None else DEFAULT_REWARD_WEIGHTS
    }
    
    metadata_path = os.path.join(save_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)