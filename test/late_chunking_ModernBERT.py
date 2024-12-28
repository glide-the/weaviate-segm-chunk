 

import os
import torch
import numpy as np 

import spacy
from spacy.tokens import Doc
from spacy.language import Language

import transformers
from transformers import AutoModel
from transformers import AutoTokenizer
 

from sentence_transformers import SentenceTransformer

def chunk_by_sentences(input_text: str, tokenizer: callable):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = []
    for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets)):
        # bert offset diff
        if token_offsets.shape[0] == i + 1:
            
            chunk_positions.append((i, int(start + 1)))
            break

        if token_id == punctuation_mark_id or (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        ):
            chunk_positions.append((i, int(start + 1)))
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations

def late_chunking(
    model_output: 'BatchEncoding', span_annotation: list, max_length=None
):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs

 

if __name__ == "__main__":
    
    document = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
 
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/mnt/ceph/develop/jiawei/model_checkpoint/ModernBERT-base', trust_remote_code=True)
    model = AutoModel.from_pretrained('/mnt/ceph/develop/jiawei/model_checkpoint/ModernBERT-base', trust_remote_code=True).to(device='cuda:0')
    seq_model = SentenceTransformer("/mnt/ceph/develop/jiawei/model_checkpoint/ModernBERT-base").to(device='cuda:0')
 
    chunks, span_annotations = chunk_by_sentences(document, tokenizer)
    print('Chunks:\n- "' + '"\n- "'.join(chunks) + '"')

    embeddings_traditional_chunking = seq_model.encode(chunks)

    # chunk afterwards (context-sensitive chunked pooling)
    inputs = tokenizer(document, return_tensors='pt').to(device='cuda:0')

    token_embeddings = model(**inputs)
    chunk_embeddings = late_chunking(token_embeddings, [span_annotations])[0]

    import numpy as np

    cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    berlin_embedding = seq_model.encode('berlin')

    for chunk, new_embedding, trad_embeddings in zip(chunks, chunk_embeddings, embeddings_traditional_chunking):
        print(f'similarity_new("berlin", "{chunk}"):', cos_sim(berlin_embedding, new_embedding))
        print(f'similarity_trad("berlin", "{chunk}"):', cos_sim(berlin_embedding, trad_embeddings))