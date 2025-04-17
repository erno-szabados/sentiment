import logging
import time
import csv
from llama_cpp import Llama

# Set logging level (optional)
logging.basicConfig(level=logging.WARNING)

# Path to your language model file
model_path = "/home/eszabados/workspace/models/SmolLM2-1.7B-Instruct-Q8_0.gguf"
#model_path = "/home/eszabados/workspace/models/SmolLM2-360M-Instruct-Q8_0.gguf"
#model_path = "/home/eszabados/workspace/models/SmolLM2-135M-Instruct-Q8_0.gguf"


# Path to your labeled sentiment data file (CSV)
data_file_path = "labeled_sentiment_data.csv"

# Define the parameter ranges for experimentation
temperatures_to_test = [0.1]
top_ps_to_test = [0.1]
top_ks_to_test = [10]
seeds_to_test = [55, 3133]  # You can add more seeds for robustness

#temperatures_to_test = [0.1]
#top_ps_to_test = [0.1]
#top_ks_to_test = [10]
#seeds_to_test = [55,3133,32,4677]  # You can add more seeds for robustness

def analyze_sentiment(llm, text):
    prompt = f"""
        Evaluate the sentiment of the following text. Classify it as positive, negative, or neutral. Example:
        Text: 'This is an example.'
        Sentiment: neutral
        Text: '{text}'
        Sentiment:"""
    output = llm(
        prompt,
        max_tokens=50,
        stop=["\n"],
        echo=True
    )
    sentiment_response = output["choices"][0]["text"].split("Sentiment:")[-1].strip().lower()
    return sentiment_response

def evaluate_accuracy(model_path, data_file_path, temperature, top_p, top_k, seed):
    llm = Llama(
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        verbose=False,
        n_ctx=8192
    )

    correct_predictions = 0
    total_samples = 0
    total_processing_time = 0

    try:
        with open(data_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                text_to_analyze = row['text'].strip()
                true_label = row['label'].strip().lower()
                if text_to_analyze:
                    start_time = time.time()
                    predicted_sentiment = analyze_sentiment(llm, text_to_analyze)
                    end_time = time.time()
                    processing_time = end_time - start_time
                    total_processing_time += processing_time
                    total_samples += 1

                    if predicted_sentiment == true_label:
                        correct_predictions += 1

        if total_samples > 0:
            accuracy = correct_predictions / total_samples
            average_processing_time = total_processing_time / total_samples
            return accuracy, average_processing_time
        else:
            return 0.0, 0.0

    except FileNotFoundError:
        print(f"Error: The file '{data_file_path}' was not found.")
        return 0.0, 0.0
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        return 0.0, 0.0

if __name__ == "__main__":
    results = []
    print("Starting parameter experimentation...")

    for temp in temperatures_to_test:
        for p in top_ps_to_test:
            for k in top_ks_to_test:
                for seed in seeds_to_test:
                    print(f"\n--- Testing Parameters: temp={temp}, top_p={p}, top_k={k}, seed={seed} ---")
                    accuracy, avg_time = evaluate_accuracy(model_path, data_file_path, temp, p, k, seed)
                    results.append({
                        'temperature': temp,
                        'top_p': p,
                        'top_k': k,
                        'seed': seed,
                        'accuracy': accuracy,
                        'average_time': avg_time
                    })
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"Average Processing Time: {avg_time:.4f} seconds")

    print("\n--- Experimentation Results ---")
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for result in sorted_results:
        print(f"Temperature: {result['temperature']}, Top_p: {result['top_p']}, Top_k: {result['top_k']}, Seed: {result['seed']}")
        print(f"  Accuracy: {result['accuracy']:.4f}, Average Time: {result['average_time']:.4f} seconds")
