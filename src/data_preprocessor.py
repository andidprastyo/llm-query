import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class DataPreprocessor:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize the data preprocessor with an embedding model.

        :param embedding_model: Sentence Transformer model for generating embeddings
        """
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))

    def load_dataset(self, filepath):
        """
        Load dataset from JSONL file (JSON Lines format).

        :param filepath: Path to the JSONL dataset
        :return: Preprocessed DataFrame
        """
        # Read the JSONL file and parse it line by line
        with open(filepath, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(data)
        return df

    def create_searchable_text(self, row):
        """
        Create a searchable text representation from a row.

        :param row: DataFrame row
        :return: Concatenated searchable text
        """
        # Customize this method based on your dataset's columns
        searchable_fields = ['nama', 'customer', 'project_nama', 'sid_tsat']
        text_parts = [str(row.get(field, '')) for field in searchable_fields]
        return ' '.join(text_parts)

    def clean_metadata_value(self, value):
        """
        Clean metadata value by converting None to appropriate default values.

        :param value: Input value
        :return: Cleaned value
        """
        if pd.isna(value) or value is None:
            return ""  # Convert None/NaN to empty string
        if isinstance(value, (int, float)):
            if pd.isna(value):  # Check for NaN/inf in numeric values
                return 0
        return str(value)  # Convert everything to string to ensure compatibility

    def preprocess(self, df):
        """
        Preprocess the DataFrame by adding searchable text and embeddings.

        :param df: Input DataFrame
        :return: Preprocessed DataFrame with embeddings
        """
        # Fill NaN values with empty strings for text columns
        text_columns = ['nama', 'customer', 'project_nama', 'sid_tsat']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Create searchable text column
        df['search_text'] = df.apply(self.create_searchable_text, axis=1)

        # Generate embeddings
        df['embeddings'] = df['search_text'].apply(lambda x: self.model.encode(x).tolist())

        return df

    def save_preprocessed_data(self, df, output_path):
        """
        Save preprocessed DataFrame to chromadb.

        :param df: Preprocessed DataFrame
        :param output_path: Path to save the preprocessed data
        """
        # Create or get a collection
        collection_name = "preprocessed_data"
        try:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Preprocessed dataset with embeddings"}
            )
        except ValueError:  # Collection already exists
            collection = self.client.get_collection(name=collection_name)
            collection.delete(ids=collection.get()['ids'])  # Clear existing data

        # Prepare data for ChromaDB with cleaned metadata
        ids = [str(i) for i in range(len(df))]
        embeddings = df['embeddings'].tolist()
        documents = df['search_text'].tolist()
        metadatas = [{
            'name': self.clean_metadata_value(row['nama']),
            'customer': self.clean_metadata_value(row['customer']),
            'project_name': self.clean_metadata_value(row['project_nama']),
            'sid_tsat': self.clean_metadata_value(row['sid_tsat'])
        } for _, row in df.iterrows()]

        # Add data to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        # Save the client's data
        self.client.persist(output_path)

    def load_preprocessed_data(self, input_path):
        """
        Load preprocessed data from chromadb.

        :param input_path: Path to the preprocessed data
        :return: Preprocessed DataFrame
        """
        # Load the persisted data
        self.client = chromadb.PersistentClient(path=input_path)
        collection = self.client.get_collection(name="preprocessed_data")
        
        # Get all data from collection
        result = collection.get()
        
        # Convert to DataFrame
        data = []
        for i in range(len(result['ids'])):
            data.append({
                'embeddings': result['embeddings'][i],
                'nama': result['metadatas'][i]['name'],
                'customer': result['metadatas'][i]['customer'],
                'project_nama': result['metadatas'][i]['project_name'],
                'sid_tsat': result['metadatas'][i]['sid_tsat'],
                'search_text': result['documents'][i]
            })
            
        return pd.DataFrame(data)

    def process_pipeline(self, input_filepath, output_filepath):
        """
        Full preprocessing pipeline.

        :param input_filepath: Input JSONL file path
        :param output_filepath: Output chromadb file path
        """
        # Load dataset
        df = self.load_dataset(input_filepath)

        # Preprocess
        processed_df = self.preprocess(df)

        # Save
        self.save_preprocessed_data(processed_df, output_filepath)

        return processed_df
    
if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    preprocessor.process_pipeline(
        '../data/v_nodelink.jsonl', 
        '../data/preprocessed_dataset.chroma'
    )