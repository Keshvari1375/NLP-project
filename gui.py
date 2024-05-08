import gradio as gr
import upload
import query

with gr.Blocks() as demo:
    gr.Markdown("# COM S 579 RAG System by Mohammad Mohammadzadeh, Mohammadreza Kiaghadi, Moones Keshvarinia")
    with gr.Tab("Upload your PDF file"):
        gr.Markdown("### Upload a PDF file to PineCone for indexing.")
        Input = gr.File(label="Select PDF", file_types=["pdf"], file_count="single")
        Output = gr.Textbox(label="Status", placeholder="PDF file is not uploaded.")
        button_upload = gr.Button("Upload the PDF file")
        button_upload.click(upload.uploader, inputs=[Input], outputs=[Output])

    with gr.Tab("Ask your question"):
        gr.Markdown("### Ask a question relevant to the uploaded PDF file")
        question = gr.Textbox(label="Question")
        result = gr.Textbox(label="Answer")
        button_query = gr.Button("Ask your question")
        button_query.click(query.query, inputs=[question], outputs=[result])

demo.launch(inbrowser=True)


