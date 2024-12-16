from src.data_preprocessor import DataPreprocessor
from src.query_engine import DatasetQueryEngine
import pandas as pd

def main():
    # Preprocessing
    preprocessor = DataPreprocessor()
    preprocessor.process_pipeline(
        './data/v_nodelink.jsonl', 
        './data/preprocessed_dataset.chroma'
    )

    # Initialize Query Engine
    query_engine = DatasetQueryEngine('./data/preprocessed_dataset.chroma')

    # Interactive Query Loop
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")

        if query.lower() == 'exit':
            break

        try:
            # Perform semantic search and LLM query
            response = query_engine.query_with_llm(query)
            print("\nResponse:", response)

            # Optional: Show top relevant rows
            print("\nTop Relevant Rows:")
            print(query_engine.semantic_search(query))

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
