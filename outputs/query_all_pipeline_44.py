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
Please extract relevant facts or answers from the following text based on the original question and the simplified sub-questions:

Original Question: {origin_question_str}

Simplified Sub-Questions:

{question_str}

Text: {context_str}

Extracted Facts or Answers:

[Answer or relevant fact for sub-question 1]
[Answer or relevant fact for sub-question 2]
[Answer or relevant fact for sub-question 3] ...
"""
QUESTION_REWRITE_PROMPT = """
Please break down the following complex multi-hop question into simpler sub-questions that can each be answered individually:

Original Question: {question_str}

Simplified Sub-Questions:

[Sub-question 1]
[Sub-question 2]
[Sub-question 3] ...
Ensure that each sub-question focuses on a single aspect or piece of information needed to answer the original question.
"""
FINALLY_ANSWER = """
Please combine the original question and the answers or facts from the simplified sub-questions to provide the most straightforward final answer, excluding any explanatory output or evidence。
Original Question: {origin_question_str}
Simplified Sub-Questions:
{question_str}
Their Answers:
{fact_str}
Final Answer:
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
    ), question_str=question_rewrite,origin_question_str=json_obj["input"]
    )
    messages = [
        ChatMessage(role="user", content=prompt),
    ]
    fact = llm.chat(messages).message.content
    fact_nodes.append(TextNode(text=fact))

    # finally_answer
    qa_template = PromptTemplate(FINALLY_ANSWER)
    prompt = qa_template.format(
        fact_str=fact,
        question_str=question_rewrite,origin_question_str=json_obj["input"]
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
