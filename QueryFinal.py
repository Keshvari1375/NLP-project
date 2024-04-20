import argparse
import os
import config
from ctransformers import AutoModelForCausalLM
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec


os.environ["PINECONE_API_KEY"] = "3de80047-64c8-4d3f-8f98-bba39eff0657"
os.environ["PINECONE_ENV"] = 'gcp-starter'
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = 'rag-demo-test'
if index_name not in pc.list_indexes().names():
    pc.create_index(name='rag-demo-test', dimension=384, metric="cosine", spec=PodSpec(environment="gcp-starter"))
    index = pc.Index(index_name)
    
index = pc.Index(index_name)
vectors = PineconeVectorStore(pinecone_index=index)
    
def query_reciever(Query, TopRanks):
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embed_model= Settings.embed_model
    model_address= "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
    #llm model
    LLM = LlamaCPP(
        model_url=model_address,
        temperature=0.1,
        max_new_tokens=384,
        context_window=3000,
        generate_kwargs={},
        verbose=False)
    embedded_queries = embed_model.get_text_embedding(Query)
    Results = index.query(vector = embedded_queries, top_k=TopRanks, include_metadata=False)
    chunks= []
    print("Query is: ", Query)
    print("Number of top ranks:", TopRanks)

    for i in Results['matches']:
        chunk =TextNode(text=i['metadata']['text'])
        chunks.append(chunk)
        print("Context:", chunk.text)
        print("****","\n")
    Ind = VectorStoreIndex(chunks)
    query_engine = Ind.as_query_engine(similarity_top_k=TopRanks, llm=LLM)
    response = query_engine.query(Query)
    print("The most relevant answer is: ")
    print(str(response))      

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Upload and index a PDF file.")
    parser.add_argument("--query", type=str, required=True, help="Path to the PDF file to be uploaded and indexed.")
    parser.add_argument('--top_k', type=int, default=5, help='top_k')
    args = parser.parse_args()
    query_reciever(args.query, args.top_k)
    