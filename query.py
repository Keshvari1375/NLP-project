from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores import PineconeVectorStore
from llama_index.retrievers import VectorIndexRetriever
import openai
import os
from pinecone.grpc import PineconeGRPC
import sys
from llama_index import VectorStoreIndex
import argparse


def query(question):
    #Initialisation
    openai.api_key = "Enter-Your-OpenAI-Key"
    pc = PineconeGRPC(api_key="Enter-Your-API-Key")
    pc_index = pc.Index("rag")
    vector_store = PineconeVectorStore(pinecone_index=pc_index)
    vec_ind = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    index_retriever = VectorIndexRetriever(index=vec_ind, similarity_top_k=8)
    result = index_retriever.retrieve(question)
    query_engine = RetrieverQueryEngine(retriever=index_retriever)
    llm_query = query_engine.query(question)
    return(llm_query.response)




#Main Part
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query")
    parser.add_argument("--query", type=str, required=True, help="Enter your question")
    args = parser.parse_args()
    print(query(args.query))
