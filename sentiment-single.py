from llama_cpp import Llama

model_path="/home/eszabados/workspace/models/SmolLM2-1.7B-Instruct-Q8_0.gguf"


llm = Llama(
    model_path=model_path,
    temperature=0.1,  # Lower temperature for more deterministic output
    top_p=0.1,        # Lower top_p also increases determinism
    top_k=5,         # Limit the number of top tokens to consider
    seed=4233,          # For reproducibility
    verbose=False     # Ensure debug output is off
)

with open('sentiment_data.txt') as data_file:
    for text in data_file:
        prompt = f"""
        Evaluate the sentiment of the following text. Classify it as positive, negative, or neutral. Example:
        Text: 'This is an example.'
        Sentiment: neutral
        Text: '{text}'
        Sentiment:"""

        output = llm(
            prompt,
            max_tokens=50,      # Adjust as needed for the expected sentiment output
            stop=["\n"],       # Stop generation at a newline
            echo=True          # Include the prompt in the output (optional)
        )

        sentiment = output["choices"][0]["text"].split("Sentiment:")[-1].strip()
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")

