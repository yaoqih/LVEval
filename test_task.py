task_list=[
    {"method":"direct","description":"直接把所有的问题和上下文放在一起问答"},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":64,"top_k":2}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":128,"top_k":2}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":256,"top_k":2}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":512,"top_k":2}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":1024,"top_k":2}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":2048,"top_k":2}},
    {"method":"llama_index_question_rewrite","description":"使用索引的方式进行问答+对问题进行重写","args":{"chunk_size":128}},
    {"method":"llama_index_question_rewrite","description":"使用索引的方式进行问答+对问题进行重写","args":{"chunk_size":256}},
    {"method":"llama_index_question_rewrite","description":"使用索引的方式进行问答+对问题进行重写","args":{"chunk_size":512}},
    {"method":"llama_index_question_rewrite","description":"使用索引的方式进行问答+对问题进行重写","args":{"chunk_size":1024}},
    {"method":"llama_index_question_rewrite","description":"使用索引的方式进行问答+对问题进行重写","args":{"chunk_size":2048}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":64,"top_k":4}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":64,"top_k":8}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":64,"top_k":16}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":128,"top_k":4}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":128,"top_k":8}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":128,"top_k":16}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":128,"top_k":32}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":128,"top_k":64}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":256,"top_k":8}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":256,"top_k":16}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":256,"top_k":32}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":512,"top_k":8}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":512,"top_k":16}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":256,"top_k":8,"text_order":True}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":256,"top_k":16,"text_order":True}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":256,"top_k":32,"text_order":True}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":512,"top_k":8,"text_order":True}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":512,"top_k":16,"text_order":True}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":256,"top_k":4}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":512,"top_k":4}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":64,"top_k":32}},
    {"method":"llama_index","description":"使用索引的方式进行问答","args":{"chunk_size":64,"top_k":64}},
    {"method":"fact_extract","description":"使用抽取后的事实回答","args":{"chunk_size":128,"top_k":16}},
    {"method":"fact_extract","description":"使用抽取后的事实回答","args":{"chunk_size":128,"top_k":32}},
    {"method":"fact_extract","description":"使用抽取后的事实回答","args":{"chunk_size":256,"top_k":16}},
    {"method":"fact_extract","description":"使用抽取后的事实回答","args":{"chunk_size":256,"top_k":8}},
    {"method":"all_pipeline","description":"综合事实抽取，问题重写，分块，重排序","args":{}},
    {"method":"all_pipeline","description":"综合事实抽取，问题重写，分块，重排序,替换掉最后冗余的再检索，使用原问题进行回答","args":{}},
    {"method":"all_pipeline","description":"综合事实抽取，问题重写，分块，重排序,替换掉最后冗余的再检索，使用原问题进行回答,使用自己的事实抽取模块","args":{}},
    {"method":"all_pipeline","description":"综合事实抽取，问题重写，分块，重排序,替换掉最后冗余的再检索，使用原问题进行回答,使用自己的事实抽取模块,全部事实一次抽取","args":{}},
    {"method":"all_pipeline","description":"综合事实抽取，问题重写，分块，重排序,替换掉最后冗余的再检索，使用原问题进行回答,使用自己的事实抽取模块,全部事实一次抽取,事实和原文一起传","args":{}},
    {"method":"all_pipeline","description":"简单的索引筛选+问答","args":{}},
    {"method":"all_pipeline","description":"简单的prompt","args":{}},
    {"method":"all_pipeline","description":"简单的prompt+原始文本","args":{}},
    {"method":"all_pipeline","description":"还原之前的prompt","args":{}},
    {"method":"all_pipeline","description":"还原之前的prompt+原始文本","args":{}},
    ]