
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an expert in the mining industry. Provide an accurate, concise answer based solely on the provided context, focusing on practical applications and industry-relevant insights. If the context does not contain enough information to answer confidently, refrain from providing an answer. Please return the response in beautiful and coherent English.

The context is as follows:

Context:
{context}

---

Question:
{question}

Answer:
"""


def format_file_reference(reference):
    if reference is None:
        return "Location: Unknown"
    # Split the reference into file path, page, and line
    file_path, page, line = reference.split(":")

    # Extract the file name from the full path
    file_name = file_path.split("\\")[-1]

    # Convert page and line numbers to be human-readable
    display_page = int(page) + 1  # Convert zero-indexed page to one-indexed
    display_line = int(line)

    # Construct and return the formatted string
    return f"""
**File Name**: `{file_name}`  
**Page no**: `{display_page}`   **Line no**: `{display_line}`
"""


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    response_text = model.invoke(prompt).content.strip()
    # in sources add the id of the document and page_content
    sources = []
    for doc, _score in results:
        sources.append({
            "id": format_file_reference(doc.metadata.get("id", None)),
            "content": doc.page_content,
            "score": round(_score*100, 2),
        })

    return {
        "response": response_text,
        "sources": sources,
    }
