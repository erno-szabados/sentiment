import logging
import time
import csv
import ollama 
import sys 

# Emotion analysis script using Ollama API


# Set logging level (optional)
logging.basicConfig(level=logging.WARNING)

# --- Ollama Configuration ---
# Name of the model available in your Ollama instance (e.g., run 'ollama list' in your terminal)
#model_name = "smollm2:1.7b" # Accuracy: 0.32
#model_name ="gemma3:1b" # Accuracy: 0.92
#model_name ="gemma3:4b" # Accuracy: 0.98
#model_name = "llama3.2:1b" # Accuracy: 0.18
#model_name="phi4-mini:latest" # Accuracy: 0.84
model_name="granite3.3:2b" # Accuracy: 0.94
# Path to your labeled data file (CSV)
data_file_path = "data/labeled_emotion_data.csv"

# Define the parameter ranges for experimentation
temperatures_to_test = [0.1]
top_ps_to_test = [0.1]
top_ks_to_test = [5]
seeds_to_test = [4233]  # You can add more seeds for robustness

emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

# --- IMPORTANT ---
# Ensure the Ollama server is running before executing this script.
# You might need to install the ollama library: pip install ollama
# ---

def analyze_emotions(model_name, text, temperature, top_p, top_k, seed):
    emotion_list = ", ".join(emotions)
    prompt = f"""
            Determine the dominant emotion of the following text. Classify it as {emotion_list}. Respond with a single word. Example:
            Text: 'I am so happy for the raise in my salary!'
            Emotion: joy
            Text: '{text}'
            Emotion:"""
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            stream=False, # Get the full response at once
            options={
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'seed': seed,
                'num_predict': 10, # Max tokens for the emotion word (generous)
                'stop': ["\n", "Text:", "Emotion:"], # Stop sequences
                'num_ctx': 4096 # Context window size (adjust if needed)
            }
        )
        # Ollama response structure gives the generated text in 'response'
        
        # Parse the response
        emotion = response['response'].split("Emotion:")[-1].strip().lower()

        return emotion

    except Exception as e:
        print(f"Error during Ollama API call: {e}")
        # Depending on the error, you might want to retry or return a default/error value
        # Check if it's a connection error specifically
        if "Connection refused" in str(e) or "Failed to connect" in str(e):
             print("\n*** Error: Could not connect to Ollama.")
             print("    Please ensure the Ollama server is running.")
             print("    You can start it by running 'ollama serve' in your terminal.")
             sys.exit(1) # Exit the script if connection fails critically
        return "error" # Return an indicator of failure


def evaluate_accuracy(model_name, data_file_path, temperature, top_p, top_k, seed):
    """
    Evaluates the accuracy of emotion analysis using Ollama over a dataset.
    """
    correct_predictions = 0
    total_samples = 0
    total_processing_time = 0
    error_count = 0

    try:
        with open(data_file_path, 'r', newline='', encoding='utf-8') as csvfile: # Added encoding
            # Detect delimiter automatically or specify if known (e.g., delimiter=';')
            try:
                dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=';,')
                csvfile.seek(0)
                reader = csv.DictReader(csvfile, dialect=dialect)
                # Verify required columns exist
                if 'text' not in reader.fieldnames or 'label' not in reader.fieldnames:
                     print(f"Error: CSV file '{data_file_path}' must contain 'text' and 'label' columns.")
                     return 0.0, 0.0, 0
            except csv.Error:
                 print(f"Error: Could not determine delimiter for CSV file '{data_file_path}'. Please ensure it's comma or semicolon separated.")
                 return 0.0, 0.0, 0
            except Exception as e:
                 print(f"Error reading CSV header: {e}")
                 return 0.0, 0.0, 0


            print(f"Processing data from {data_file_path}...")
            count = 0
            for row in reader:
                # Basic check for valid row structure
                if 'text' not in row or 'label' not in row:
                    print(f"Warning: Skipping invalid row: {row}")
                    continue

                text_to_analyze = row.get('text', '').strip()
                true_label = row.get('label', '').strip().lower()

                if text_to_analyze and true_label in emotions:
                    start_time = time.time()
                    predicted_emotion = analyze_emotions(model_name, text_to_analyze, temperature, top_p, top_k, seed)
                    end_time = time.time()

                    if predicted_emotion == "error":
                        error_count += 1
                        continue # Skip this sample if API call failed

                    processing_time = end_time - start_time
                    total_processing_time += processing_time
                    total_samples += 1

                    if predicted_emotion == true_label:
                        correct_predictions += 1

                    count += 1
                    if count % 50 == 0: # Print progress
                        print(f"  Processed {count} samples...")


        if total_samples > 0:
            accuracy = correct_predictions / total_samples
            average_processing_time = total_processing_time / total_samples
            print(f"Finished processing {total_samples} valid samples.")
            if error_count > 0:
                print(f"Encountered {error_count} errors during API calls.")
            return accuracy, average_processing_time, error_count
        else:
            print("No valid samples found in the CSV file.")
            return 0.0, 0.0, error_count

    except FileNotFoundError:
        print(f"Error: The file '{data_file_path}' was not found.")
        return 0.0, 0.0, 0
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0

if __name__ == "__main__":
    results = []
    print("Starting parameter experimentation with Ollama...")
    print(f"Using Ollama model: {model_name}")
    print(f"Using data file: {data_file_path}")
    print("-" * 30)

    # --- Pre-flight check for Ollama connection ---
    try:
        print("Checking connection to Ollama server...")
        ollama.list() # Simple command to check connectivity and list models
        print("Ollama connection successful.")
        # You could add a check here to ensure model_name exists in ollama.list() if desired
    except Exception as e:
         print(f"\n*** Error: Could not connect to Ollama: {e}")
         print("    Please ensure the Ollama server is running.")
         print("    You can start it by running 'ollama serve' in your terminal.")
         sys.exit(1)
    print("-" * 30)
    # ---

    for temp in temperatures_to_test:
        for p in top_ps_to_test:
            for k in top_ks_to_test:
                for seed in seeds_to_test:
                    print(f"\n--- Testing Parameters: temp={temp}, top_p={p}, top_k={k}, seed={seed} ---")
                    accuracy, avg_time, errors = evaluate_accuracy(model_name, data_file_path, temp, p, k, seed)
                    results.append({
                        'temperature': temp,
                        'top_p': p,
                        'top_k': k,
                        'seed': seed,
                        'accuracy': accuracy,
                        'average_time': avg_time,
                        'errors': errors
                    })
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"Average Processing Time: {avg_time:.4f} seconds")
                    if errors > 0:
                        print(f"API Errors during run: {errors}")

    print("\n--- Experimentation Results ---")
    # Sort by accuracy (descending), then by average time (ascending) as a tie-breaker
    sorted_results = sorted(results, key=lambda x: (x['accuracy'], -x['average_time']), reverse=True)
    for result in sorted_results:
        print(f"Temp: {result['temperature']}, Top_p: {result['top_p']}, Top_k: {result['top_k']}, Seed: {result['seed']}")
        print(f"  Accuracy: {result['accuracy']:.4f}, Avg Time: {result['average_time']:.4f}s, Errors: {result['errors']}")

    print("\nExperimentation finished.")
