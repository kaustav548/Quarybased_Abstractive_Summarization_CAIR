from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
import torch
import re

class ModelABSQ:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    def modelABSQ(paragraph,query,Max_length,Min_length):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        model = PegasusForConditionalGeneration.from_pretrained("my_finetuned_model")
        tokenizer = PegasusTokenizer.from_pretrained("my_finetuned_model")
        sentence_model = SentenceTransformer('my_sentence_model')
        paragraph = str(str(str(paragraph)))
        query = str(query)
        # Clean and normalize text
        clean_text = ' '.join(paragraph.split())

        # Split into sentences
        sentences = sent_tokenize(clean_text)

        # Clean sentences but preserve structure
        cleaned_sentences = []
        for sentence in sentences:
            # Remove special characters but keep periods and basic punctuation
            cleaned = re.sub(r'[^\w\s.,!?;:]', ' ', sentence)
            cleaned = ' '.join(cleaned.split())  # Remove extra spaces
            if len(cleaned.strip()) > 10:  # Filter very short sentences
                cleaned_sentences.append(cleaned)

        print(f"Document split into {len(cleaned_sentences)} sentences")
        for i, sent in enumerate(cleaned_sentences):
            print(f"{i+1}. {sent}")
        
        # Process sentences and query for embedding
        processed_sentences = [ModelABSQ.lemmatize_text(sent) for sent in cleaned_sentences]
        processed_query = ModelABSQ.lemmatize_text(query)

        print(f"Original query: {query}")
        print(f"Processed query: {processed_query}")
        print(f"\nProcessed sentences:")
        for i, sent in enumerate(processed_sentences):
            print(f"{i+1}. {sent}")

        # Generate embeddings
        print("Generating embeddings...")
        sentence_embeddings = sentence_model.encode(processed_sentences)
        query_embedding = sentence_model.encode([processed_query])

        # Calculate similarities
        similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]

        # Set parameters
        top_k = 5
        threshold = 0.3

        # Filter by threshold and get top-k
        relevant_indices = []
        for i, sim in enumerate(similarities):
            if sim.item() > threshold:
                relevant_indices.append((i, sim.item()))

        # Sort by similarity and take top-k
        relevant_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in relevant_indices[:top_k]]

        # Return sentences in original order
        top_indices.sort()
        relevant_sentences = [cleaned_sentences[i] for i in top_indices]
        similarity_scores = [similarities[i].item() for i in top_indices]

        print(f"Found {len(relevant_sentences)} relevant sentences")
        print(f"Similarity scores: {[f'{s:.3f}' for s in similarity_scores]}")
        print("\nRelevant sentences:")
        for i, (sent, score) in enumerate(zip(relevant_sentences, similarity_scores)):
            print(f"{i+1}. (Score: {score:.3f}) {sent}")
    
        if not relevant_sentences:
            summary = "No relevant information found for the given query."
        else:
            # Combine relevant sentences
            input_text = " ".join(relevant_sentences)

            # Add query context to guide summarization
            prompt = f"Query: {query}\nText: {input_text}"

            print("Generating summary...")
            print(f"Input length: {len(prompt.split())} words")

            # Tokenize with proper truncation
            inputs = tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # Generate summary with conservative parameters to reduce hallucination
            with torch.no_grad():
                summary_ids = model.generate(
                    **inputs,
                    max_length=Max_length,      # Maximum summary length
                    min_length=Min_length,       # Minimum summary length
                    num_beams=4,         # Beam search for better quality
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # Avoid repetition
                    do_sample=False,     # Deterministic generation
                    temperature=1.0,     # Conservative temperature
                    repetition_penalty=1.2,  # Penalize repetition
                    length_penalty=1.0,  # Neutral length penalty
                )

            # Decode summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Post-process to remove artifacts
            summary = summary.replace("<pad>", "").strip()
            summary = summary.replace("<n>", " ").strip()
        print(f"\nGenerated Summary:")
        print(f"'{summary}'")
        return summary

    def lemmatize_text(text):
        words = word_tokenize(text.lower())
        lemmatized_words = [
            ModelABSQ.lemmatizer.lemmatize(word)
            for word in words
            if word.isalnum() and word not in ModelABSQ.stop_words
        ]
        return ' '.join(lemmatized_words)



