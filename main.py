import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )



file_path = "data/ai_notes.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"File not found: {file_path}\n"
        "Make sure data/ai_notes.txt exists."
    )

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Document loaded successfully.")



splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.create_documents([text])

print(f"Number of chunks created: {len(chunks)}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")



vectorstore = Chroma(
    collection_name="ai_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Check whether documents are already stored
existing_data = vectorstore.get()

if len(existing_data["ids"]) == 0:

    # Create unique IDs for chunks
    chunk_ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    vectorstore.add_documents(
        documents=chunks,
        ids=chunk_ids
    )

    print("Documents added to ChromaDB.")

else:

    print(
        f"ChromaDB already contains "
        f"{len(existing_data['ids'])} documents."
    )


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)

print("Retriever created successfully.")



llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key
)

print("Groq LLM initialized successfully.")



prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the question using ONLY the information
provided in the context.

If the answer is not present in the context,
say:

"I don't know based on the provided documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)



question = input("\nAsk a question: ")



docs = retriever.invoke(question)

print("\nRetrieved documents:")

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:300])



context = "\n\n".join(
    doc.page_content
    for doc in docs
)



prompt_value = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


response = llm.invoke(prompt_value)


print("ANSWER")

print(response.content)