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

FACT_EXTRACT_TEMPLATE = """Task: Extract relevant information from the given context to answer the following complex question and its sub-questions.
**Task Requirements:**
Please extract all factual claims related to the complex question and its sub-questions from the above contextual text. Note that only facts are needed, without any analysis or interpretation. If a sub-question is not mentioned in the context, please mark it as "Not mentioned".
### Example
```
**Complex Question Description:**
How can the internal communication efficiency of a company be improved?

**Sub-question List:**
1. What are the current communication problems in the company?
2. How satisfied are the employees with the communication efficiency?

**Contextual Text:**
The main communication issues within the company include untimely information transmission and a lack of coordination between departments. Employees generally believe that communication efficiency is low, as reflected in a recent satisfaction survey. The company currently uses email and internal instant messaging tools. Other companies seem to have better results with video conferencing and collaboration platforms. Some potential improvement solutions include regular cross-departmental meetings and the introduction of new communication tools.

**Answer Format:**
1. **What are the current communication problems in the company?** Untimely information transmission, lack of coordination between departments.
2. **How satisfied are the employees with the communication efficiency?** Employees generally believe that communication efficiency is low.
```
**Complex Question Description:**
{origin_question_str}
**Sub-question List:**
{question_str}
**Contextual Text:**
{context_str}
**Answer**:
"""
QUESTION_REWRITE_PROMPT = """
You are an expert in question analysis and decomposition. Your task is to break down complex, multi-hop questions into a series of simpler sub-questions that, when answered in sequence, will lead to the answer of the original question. Please follow these guidelines:
1. Carefully analyze the given multi-hop question.
2. Identify the main components and logical steps required to answer the question.
3. Break down the question into 2-5 sub-questions, depending on the complexity.
4. Ensure that the sub-questions are:
a. Clear and concise
b. Logically ordered
c. Answerable with a single piece of information
d. Progressively building towards the answer to the original question
5. Only focus on decomposing the question, do not provide the answers to the sub-questions.
6. Do not include any additional information or context beyond the question itself.
7. Do not include any explanations or reasoning for the sub-questions.
Output format:
----------------
1. [First sub-question]
2. [Second sub-question]
2. [Third sub-question]
......
Please decompose the following multi-hop question:
{question_str}
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
        embed_model.get_query_embedding(json_obj["input"]+question_rewrite),
        top_k,
    )
    nodes_filtered = [nodes[i] for i in sorted(res)]

    # fact_extract
    # fact_nodes = []
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
    # fact_nodes.append(TextNode(text=fact))

    # finally_answer
    qa_template = PromptTemplate(FINALLY_ANSWER)
    prompt = qa_template.format(
        fact,
        query_str=json_obj["input"],
    )
    messages = [
        ChatMessage(role="user", content=prompt),
    ]
    answer = llm.chat(messages).message.content
    return answer


if __name__ == "__main__":
    import json
    json_obj=json.loads(open("./hotpotwikiqa_mixup/hotpotwikiqa_mixup_16k.jsonl",encoding='utf-8').readline())
    print(query(json_obj, 31))
