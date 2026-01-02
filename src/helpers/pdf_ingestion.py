import os
import json
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter


def ingest_pdfs(pdf_directory: str, output_json_path: str):
    """
    Scans a directory for PDFs, converts them to Markdown, splits them by headers,
    and saves them to a JSON corpus file compatible with HippoRAG.

    Args:
        pdf_directory (str): The folder containing PDF files.
        output_json_path (str): The path where the corpus.json should be saved.
    """
    corpus = []
    doc_id_counter = 0

    if not os.path.exists(pdf_directory):
        print(f"Error: Directory {pdf_directory} does not exist.")
        return

    print(f"Scanning {pdf_directory} for PDFs...")

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            print(f"  - Parsing {filename}...")

            try:
                md_text = pymupdf4llm.to_markdown(filepath)

                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                docs = markdown_splitter.split_text(md_text)

                # Format for HippoRAG (Title + Text + Idx)
                for doc in docs:
                    header_context = f"{doc.metadata.get('Header 1', '')} > {doc.metadata.get('Header 2', '')}"
                    full_content = f"Context: {header_context}\n\n{doc.page_content}"

                    corpus.append({
                        "title": f"{filename} - Chunk {doc_id_counter}",
                        "text": full_content,
                        "idx": doc_id_counter
                    })
                    doc_id_counter += 1
            except Exception as e:
                print(f"    Error processing {filename}: {e}")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=4)

    print(f"Successfully processed {len(corpus)} chunks into {output_json_path}")