import pandas as pd
import numpy as np
import requests
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import chromadb

class DatasetQueryEngine:
    def __init__(self, input_path, embedding_model='all-MiniLM-L6-v2', 
                 llm_model='llama3.2:1b', 
                 ollama_base_url='http://localhost:11434/api'):
        """
        Initialize the query engine with preprocessed data and models.

        :param input_path: Path to the preprocessed data
        :param embedding_model: Sentence Transformer model for embeddings
        :param llm_model: Ollama LLM model to use
        :param ollama_base_url: Base URL for Ollama API
        """
        preprocessor = DataPreprocessor()
        self.df = preprocessor.load_preprocessed_data(input_path)
        self.embedder = SentenceTransformer(embedding_model)
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url

    def semantic_search(self, query, top_k=5):
        """
        Perform semantic search on the dataset.

        :param query: Search query string
        :param top_k: Number of top results to return
        :return: Top K most similar rows
        """
        # Generate embedding for query
        query_embedding = self.embedder.encode(query)

        # Calculate cosine similarity
        similarities = self.df['embeddings'].apply(
            lambda x: 1 - cosine(query_embedding, x)
        )

        # Get top K results
        top_results = similarities.nlargest(top_k)
        return self.df.loc[top_results.index]

    def query_with_llm(self, query):
        """
        Query the dataset using LLM with semantic search context.

        :param query: Natural language query
        :return: LLM-generated response
        """
        # Perform semantic search first
        relevant_rows = self.semantic_search(query)

        # Prepare context for LLM
        context = relevant_rows.to_json(orient='records')

        # Construct prompt for Ollama
        full_prompt = f"""
        Context: {context}

        Query: {query}

        Please analyze the context and provide a precise answer to the query. 
        Be concise and focus on the most relevant information.
        """

        try:
            # Query Ollama API directly
            response = requests.post(
                f'{self.ollama_base_url}/chat',
                json={
                    'model': self.llm_model,
                    'messages': [
                        {
                            'role': 'system', 
                            'content': 'You are a helpful data analysis assistant.'
                        },
                        {
                            'role': 'user', 
                            'content': full_prompt
                        }
                    ],
                    'stream': False
                }
            )

            # Check if request was successful
            response.raise_for_status()

            # Parse and return response
            return response.json()['message']['content']

        except requests.RequestException as e:
            return f"Error querying Ollama API: {str(e)}"
        except KeyError as e:
            return f"Error parsing Ollama API response: {str(e)}"

    def count_by_attribute(self, attribute):
        """
        Count occurrences of unique values in a given attribute.

        :param attribute: Column name to count
        :return: Value counts for the attribute
        """
        return self.df[attribute].value_counts()

    def filter_by_condition(self, column, condition):
        """
        Filter DataFrame by a given condition.

        :param column: Column to apply condition on
        :param condition: Lambda function defining the condition
        :return: Filtered DataFrame
        """
        return self.df[self.df[column].apply(condition)]

    def get_ollama_models(self):
        """
        Retrieve available Ollama models.

        :return: List of available models
        """
        try:
            response = requests.get(f'{self.ollama_base_url}/tags')
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return f"Error retrieving Ollama models: {str(e)}"
            
