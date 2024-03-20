
import sys
import os
import argparse
from PyPDF2 import PdfReader
import torch
from pinecone import Pinecone, PodSpec
import pdfminer
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , AutoModel
import numpy as np

# Initialize Pinecone

pc = Pinecone(api_key="3de80047-64c8-4d3f-8f98-bba39eff0657")
name_of_index = 'rag-demo-test'



if name_of_index not in pc.list_indexes().names():
    pc.create_index(name=name_of_index, dimension=384, metric="cosine", spec=PodSpec(environment="gcp-starter"))

#If the name already exists:
index_name = pc.Index(name_of_index)



def text_extractor(pdf_dir):
    text = extract_text(pdf_dir)
    print("Text has been extracted from PDF file")
    return text

# Clean text and remove page number
def text_preprocessor(text):
    processed_text = text.replace('Page | ', '')
    print("Text has been cleaned")
    return processed_text


#Chunk the text with chunk size of 500
def text_chunker(text, chunk_size=400):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    print("Text has been split into chunks of size 400")
    return chunks




def text_embedder(chunk):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.squeeze().tolist()




# Main RAG workflow
def main_rag_workflow(pdf_dir):
    text = text_extractor(pdf_dir)
    processed_text = text_preprocessor(text)
    chunks = text_chunker(processed_text)
    for num, chunk in enumerate(chunks):
        embedding = text_embedder(chunk)
        index_name.upsert(vectors=[(str(num), embedding)])
    print("Text has been embeded")


    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload and index a PDF file.")
    parser.add_argument("--pdf_file", type=str, required=True, help="Path to the PDF file to be uploaded and indexed.")
    
    args = parser.parse_args()
    # Function usage
    main_rag_workflow(args.pdf_file)
    print(f"The PDF file has been uploaded and indexed {args.pdf_file}.")





