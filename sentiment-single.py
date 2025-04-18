# sentiment-single-ollama.py
import ollama
import sys

# --- Ollama Configuration ---
# Name of the model available in your Ollama instance (e.g., run 'ollama list')
#model_name = "smollm2:1.7b"
#model_name ="gemma3:1b"
#model_name ="gemma3:4b"
#model_name = "llama3.2:1b"
model_name="granite3.3:2b"

# Path to your sentiment data file (one text per line)
data_file_path = 'sentiment_data.txt'

# --- Generation Parameters ---
temperature = 0.1
top_p = 0.1
top_k = 5 
seed = 4233 
max_output_tokens = 32 # Max tokens for the sentiment word (positive/negative/neutral)
stop_sequences = ["\n"] # Stop sequences

# --- IMPORTANT ---
# Ensure the Ollama server is running before executing this script.
# You might need to install the ollama library: pip install ollama
# ---

# --- Pre-flight check for Ollama connection ---
try:
    print("Checking connection to Ollama server...")
    ollama.list() # Simple command to check connectivity
    print(f"Ollama connection successful. Using model: {model_name}")
except Exception as e:
     print(f"\n*** Error: Could not connect to Ollama: {e}")
     print("    Please ensure the Ollama server is running.")
     print("    You can start it by running 'ollama serve' in your terminal.")
     sys.exit(1)
print("-" * 30)
# ---

try:
    with open(data_file_path, 'r', encoding='utf-8') as data_file:
        print(f"Reading texts from: {data_file_path}")
        for line in data_file:
            text = line.strip() # Remove leading/trailing whitespace, including newline
            if not text: # Skip empty lines
                continue

            prompt = f"""
                    Evaluate the sentiment of the following text. Classify it as positive, negative, or neutral. Respond with a single word. Example:
                    Text: 'This is an example.'
                    Sentiment: neutral
                    Text: '{text}'
                    Sentiment:"""
            try:
                # Call the Ollama API
                response = ollama.generate(
                    model=model_name,
                    prompt=prompt,
                    stream=False, # Get the full response at once
                    options={
                        'temperature': temperature,
                        'top_p': top_p,
                        'top_k': top_k,
                        'seed': seed,
                        'num_predict': max_output_tokens,
                        'stop': stop_sequences,
                        'num_ctx': 512 # Context window size (adjust if needed) - Optional
                    }
                )

                # Parse the response
                sentiment = response['response'].split("Sentiment:")[-1].strip().lower()

                print(f"Text: {text}")
                print(f"Sentiment: {sentiment}")
                print("---") # Separator between entries

            except Exception as e:
                print(f"Error processing text '{text}': {e}")
                # Check for connection errors specifically if needed (though initial check helps)
                if "Connection refused" in str(e) or "Failed to connect" in str(e):
                    print("\n*** Error: Lost connection to Ollama server during processing.")
                    print("    Please ensure the Ollama server is still running.")
                    sys.exit(1)
                # Continue to the next line on other errors
                print("---")


except FileNotFoundError:
    print(f"Error: The file '{data_file_path}' was not found.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

print("\nFinished processing all texts.")
