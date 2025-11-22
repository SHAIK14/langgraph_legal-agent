from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, Table, Image
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.messages import HumanMessage

load_dotenv()

# Initialize models
model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
vision_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

print("📄 Parsing PDF...")
path = "data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
elements = partition_pdf(filename=path, strategy="fast")
print(f"✅ Found {len(elements)} elements\n")


def chunk_by_title(elements):
    """Group elements into semantic chunks based on titles"""
    chunks = []
    current_chunk = None

    for ele in elements:
        if isinstance(ele, Title):
            # Save previous chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # Start new chunk
            current_chunk = {
                "title": ele.text,
                "elements": [ele],
                "page": getattr(ele.metadata, 'page_number', 0)
            }
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk["elements"].append(ele)

    # Save last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def process_chunk(chunk):
    """Convert a chunk into a LangChain Document with table and image summaries"""
    title = chunk['title']
    elements = chunk['elements']
    
    # Collect text, tables, images
    text_parts = []
    table_summaries = []
    table_htmls = []
    image_descriptions = []
    image_base64 = []

    for elem in elements:
        if isinstance(elem, Table):
            # Summarize table with LLM
            try:
                table_html = getattr(elem.metadata, 'text_as_html', elem.text)
                
                # Skip if table is empty or too short
                if not table_html or len(table_html.strip()) < 10:
                    print(f"  ⏭️  Skipping empty table")
                    continue
                
                prompt = f"Summarize this table in 1-2 sentences:\n{table_html[:1000]}"  # Limit to 1000 chars
                response = model.invoke(prompt)
                summary = response.content if response.content else "Table data"
                
                table_summaries.append(summary)
                table_htmls.append(table_html)
                print(f"  ✅ Table: {summary[:60]}...")
            except Exception as e:
                print(f"  ❌ Error summarizing table: {e}")
            
        elif isinstance(elem, Image):
            # Describe image with GPT-4 Vision
            try:
                image_b64 = getattr(elem.metadata, 'image_base64', None)
                
                if image_b64:
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": "Describe this image in 2-3 sentences."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                        ]
                    )
                    
                    description = vision_model.invoke([message]).content
                    image_descriptions.append(description)
                    image_base64.append(image_b64)
                    print(f"  🖼️  Image: {description[:60]}...")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
        else:
            # Regular text
            if elem.text and elem.text.strip():
                text_parts.append(elem.text)
    
    # Combine into page_content
    content = f"Section: {title}\n\n"
    content += "\n\n".join(text_parts)
    
    if table_summaries:
        content += "\n\nTables:\n"
        for i, summary in enumerate(table_summaries, 1):
            content += f"Table {i}: {summary}\n"
    
    if image_descriptions:
        content += "\n\nImages:\n"
        for i, desc in enumerate(image_descriptions, 1):
            content += f"Image {i}: {desc}\n"
    
    return Document(
        page_content=content,
        metadata={
            'title': title,
            'page': chunk['page'],
            'num_tables': len(table_summaries),
            'num_images': len(image_descriptions)
            # Note: Removed table_htmls and image_base64s - ChromaDB doesn't support lists
        }
    )


# Main execution
print("📚 Chunking by title...")
chunks = chunk_by_title(elements)
print(f"✅ Created {len(chunks)} chunks\n")

print(f"🔄 Processing first 10 chunks...\n")
documents = []

for i, chunk in enumerate(chunks[:10]):
    print(f"[{i+1}/10] {chunk['title'][:50]}...")
    doc = process_chunk(chunk)
    documents.append(doc)

print(f"\n✅ Created {len(documents)} documents!")

# Stats
total_tables = sum(doc.metadata['num_tables'] for doc in documents)
total_images = sum(doc.metadata['num_images'] for doc in documents)
print(f"\n📊 Tables: {total_tables}, Images: {total_images}")

# Store in ChromaDB
print(f"\n💾 Storing in ChromaDB...")
embeddings = OpenAIEmbeddings()

# Filter out complex metadata (lists, dicts, etc.)
filtered_docs = filter_complex_metadata(documents)

vectorstore = Chroma.from_documents(
    documents=filtered_docs,
    embedding=embeddings,
    persist_directory="./chroma_multimodal_db"
)
print(f"✅ Stored!")

# Query
print(f"\n🔍 Testing query...")
query = "What is the Transformer architecture?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nQuery: {query}\nRetrieved {len(results)} chunks\n")

# Generate answer
context = "\n\n".join([doc.page_content for doc in results])
prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
answer = model.invoke(prompt)
print(f"\n✅ Answer:\n{answer.content}\n")












