from pathlib import Path
import os
from pinecone.grpc import PineconeGRPC
from llama_index.readers import PDFReader
import sys
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings import OpenAIEmbedding
from llama_index.ingestion import IngestionPipeline
from llama_index.vector_stores import PineconeVectorStore
import argparse


#Initialisation
pc = PineconeGRPC(api_key="Enter-Your-Pinecone-API-Key")
pc_index = pc.Index("rag")
vector_store = PineconeVectorStore(pinecone_index=pc_index)
embed_model = OpenAIEmbedding(api_key="Enter-Your-OpenAI-Key")

#Main Part
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload and index a PDF file.")
    parser.add_argument("--pdf_file", type=str, required=True, help="Path to the PDF file to be uploaded and indexed.")
    args = parser.parse_args()
    PL = IngestionPipeline(transformations=[SemanticSplitterNodeParser(buffer_size=1,breakpoint_percentile_threshold=95,embed_model=embed_model,),embed_model,],vector_store=vector_store)
    texts = PDFReader().load_data(file=Path(args.pdf_file))
    PL.run(documents=texts)
    print(f"The PDF file has been uploaded and indexed {args.pdf_file}.")

