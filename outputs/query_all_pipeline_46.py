from test_task import task_list
from llama_index.core import Document
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from scipy.spatial.distance import cosine

embed_model = OpenAIEmbedding(
    api_key="sk-...", api_base="http://172.10.10.162:18001/v1"
)
llm = OpenAI(
    api_key="sk-...",
    api_base="http://172.10.10.162:18005/v1",
    temperature=0,
    max_tokens=12800,
    timeout=600,
)

FACT_EXTRACT_TEMPLATE = """
You are an AI assistant specialized in extracting factual claims from text. Your task is to analyze the given draft response and extract all factual statements that require validation. Please follow these guidelines:
    1. Read the provided question and draft response carefully.
    2. Identify all factual claims present in the draft response.
    3. Break down complex statements into simpler, atomic factual claims.
    4. Present each extracted claim as a separate item in a list.
    5. Ensure that each claim is a complete, standalone statement.
    6. Do not include opinions, speculations, or subjective statements.
    7. Focus only on extracting claims directly related to the question and response.
    8. Write one per line.
    Here's an example to guide you:

    Question: When is Frédéric Chopin's father's birthday?
    Draft Response: Frédéric Chopin's father is Nicolas Chopin, he was born on June 17, 1771.

    Extracted Claims:
    1. Frédéric Chopin's father is Nicolas Chopin.
    2. Nicolas Chopin was born on June 17, 1771.

    Now, please analyze the following question and draft response, then extract the factual claims,If the content provided does not contain any factual claims that directly answer the question, please just answer: No! without any explanation:

    Question: {question_str}
    Draft Response: {context_str}

    Extracted Claims:
"""
QUESTION_REWRITE_PROMPT = """
"You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate 3 search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {question_str}\n"
    "Queries:\n
"""
FINALLY_ANSWER = """
    Answer the question based on the given passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context_str}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {query_str}\nAnswer:
"""

def find_top_k_similar(text_embeddings, question_embedding, k=5):
    def cosine_similarity(vec1, vec2):
        return 1 - cosine(vec1, vec2)

    similarities = [
        cosine_similarity(question_embedding, text_embedding)
        for text_embedding in text_embeddings
    ]
    sorted_indices = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    top_k_indices = [idx for idx, _ in sorted_indices[:k]]
    return top_k_indices


def query(json_obj, task_index):
    # chunked
    documents = [Document(text=json_obj["context"])]
    chunk_size = 256
    if "chunk_size" in task_list[task_index]["args"]:
        chunk_size = task_list[task_index]["args"]["chunk_size"]
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=10)
    text_chunks = []
    for doc_idx, page in enumerate(documents):
        page_text = page.text
        cur_text_chunks = text_splitter.split_text(page_text)
        text_chunks.extend(cur_text_chunks)
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        nodes.append(node)

    # question_rewrite
    qa_template = PromptTemplate(QUESTION_REWRITE_PROMPT)
    prompt = qa_template.format(question_str=json_obj["input"])
    messages = [
        ChatMessage(role="user", content=prompt),
    ]
    question_rewrite = llm.chat(messages).message.content

    # node_filter
    top_k = 8
    if "top_k" in task_list[task_index]["args"]:
        top_k = task_list[task_index]["args"]["top_k"]
    res = find_top_k_similar(
        [embed_model.get_text_embedding(node.text) for node in nodes],
        embed_model.get_query_embedding(json_obj["input"]),
        top_k,
    )
    nodes_filtered = [nodes[i] for i in sorted(res)]

    # fact_extract
    fact_nodes = []
    # for node in nodes_filtered:
    fact_extract_template = PromptTemplate(FACT_EXTRACT_TEMPLATE)
    prompt = fact_extract_template.format(
        context_str="\n".join(
        [
            f"content {index}:\n{node.text}\n"
            for index, node in enumerate(nodes_filtered)
        ]
    ), question_str=question_rewrite
    )
    messages = [
        ChatMessage(role="user", content=prompt),
    ]
    fact = llm.chat(messages).message.content
    fact_nodes.append(TextNode(text=fact))

    # finally_answer
    qa_template = PromptTemplate(FINALLY_ANSWER)
    prompt = qa_template.format(
        context_str=fact,query_str=json_obj["input"]
    )
    messages = [
        ChatMessage(role="user", content=prompt),
    ]
    answer = llm.chat(messages).message.content
    return answer


if __name__ == "__main__":
    import json
    json_obj=json.loads(open("./hotpotwikiqa_mixup/hotpotwikiqa_mixup_16k.jsonl",encoding='utf-8').read().split('\n')[3])
    print(query(json_obj, 40))
