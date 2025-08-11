from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data.translation import translation_dict

english_sentences = list(translation_dict.values()) #get english sentences

#load model
sequence_encoder = SentenceTransformer('all-mpnet-base-v2')

all_embeddings = sequence_encoder.encode(english_sentences) #get embeddings for sentences

#assume our labels is our vocab list, the input 'target_embedding' is either a custom text input or the predicted models embedding. We compute a cosine similarity
#between the target and all current embeddings then return the actual mapped embedding from our vocab list as the 'nearest' conceptual expression of the sentence intent. The more variation of similar
#phrases in our vocab storage, the more types of words/sentences that the model can use.... but this is a limitation in actual implementation because it relies on hardcoded variations of sentences with increasing
#capacity, it doesnt naturally let the model select tokens or words to generate coherent sentences
class Decoder:
    def __init__(self, sentences, embeddings):
        self.sentences = sentences
        self.embeddings = embeddings

    def decode(self, target_embedding, use_mse=False):
        if use_mse:
            #compute mse distances (lower = better match)
            mse_distances = np.mean((self.embeddings - target_embedding) ** 2, axis=1)
            best_match_idx = np.argmin(mse_distances)
            mse_distance = mse_distances[best_match_idx]
            #convert mse to confidence score (invert since lower mse = higher confidence)
            conf = 1.0 / (1.0 + mse_distance)
        else:
            #original cosine similarity (higher = better match)
            similarities = cosine_similarity([target_embedding], self.embeddings)[0]
            best_match_idx = np.argmax(similarities)
            conf = similarities[best_match_idx]

        return self.sentences[best_match_idx], conf, best_match_idx, self.embeddings[best_match_idx]
    
    def compare_texts(self, input_text, target_text, use_mse=False):
        #encode both texts to embeddings
        input_embedding = sequence_encoder.encode([input_text])[0]
        target_embedding = sequence_encoder.encode([target_text])[0]
        
        if use_mse:
            #compute mse distance between embeddings
            mse_distance = np.mean((input_embedding - target_embedding) ** 2)
            return mse_distance
        else:
            #compute cosine similarity between embeddings
            similarity = cosine_similarity([input_embedding], [target_embedding])[0][0]
            return similarity
    
decoder = Decoder(english_sentences, all_embeddings)

def test_custom_input(input_text, use_mse=False):
    print(f"Input: '{input_text}'")
    
    # Encode the input text
    input_embedding = sequence_encoder.encode([input_text])[0]
    
    # Decode using custom vocabulary
    closest_sentence, confidence, match_idx, embedding = decoder.decode(input_embedding, use_mse=use_mse)
    
    print(f"Closest match: '{closest_sentence}' with embedding: {embedding}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Match index: {match_idx}")
    print("-" * 50)
    
    return closest_sentence, confidence

def test_similarity_comparison(input_text, target_text, use_mse=False):
    #test direct similarity between two text inputs
    similarity_score = decoder.compare_texts(input_text, target_text, use_mse=use_mse)
    
    metric_name = "MSE distance" if use_mse else "Cosine similarity"
    print(f"Input: '{input_text}'")
    print(f"Target: '{target_text}'")
    print(f"{metric_name}: {similarity_score:.4f}")
    
    if use_mse:
        print(f"Converted confidence: {1.0 / (1.0 + similarity_score):.4f}")
    
    print("-" * 50)
    
    return similarity_score

#example usage
# test_custom_input("Child school", use_mse=False)

# test_similarity_comparison("child goes to school", "Child goes to kindergarten")
test_similarity_comparison("CanadaCanadaCanadaCanadaCanadaCanada", "Canada is really a good place") 
# test_similarity_comparison(" ", "the")