import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

load_dotenv(override=True)

def make_prompt(ddl, question, query=''):
    """ SQL 생성 프롬프트 """
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
{ddl}

### Question:
{question}

### SQL:
{query}"""
    return prompt

def make_requests_for_gpt_evaluation(df, filename, dir='requests'):
    """평가를 위한 jsonl 파일 생성 함수"""
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)
    prompts = []
    for idx, row in df.iterrows():
        prompts.append("""Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" + f"""
DDL: {row['context']}
Question: {row['question']}
gt_sql: {row['answer']}
gen_sql: {row['gen_sql']}"""
                       )
    jobs = [{"model": "gpt-4-turbo-preview", "response_format" : { "type": "json_object" }, "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]
    with open(Path(dir, filename), "w") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")

def change_jsonl_to_csv(input_file, output_file, prompt_column="prompt", response_column="response"):
    """결과 jsonl을 csv로 바꿔주는 함수"""
    prompts = []
    responses = []
    with open(input_file, 'r') as json_file:
        for data in json_file:
            prompts.append(json.loads(data)[0]['messages'][0]['content'])
            responses.append(json.loads(data)[1]['choices'][0]['message']['content'])

    df = pd.DataFrame({prompt_column: prompts, response_column: responses})
    df.to_csv(output_file, index=False)
    return df

def make_inference_pipeline(model_id):
    """추론 파이프라인 생성"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

model_id = 'beomi/Yi-Ko-6B'
hf_pipe = make_inference_pipeline(model_id)

example = """당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
CREATE TABLE players (
  player_id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  date_joined DATETIME NOT NULL,
  last_login DATETIME
);

### Question:
사용자 이름에 'admin'이 포함되어 있는 계정의 수를 알려주세요.

### SQL:
"""

# df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
# df = df.to_pandas()
# for idx, row in df.iterrows():
#     prompt = make_prompt(row['context'], row['question'])
#     df.loc[idx, 'prompt'] = prompt
# # sql 생성
# gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False, return_full_text=False, max_length=512, truncation=True)
# gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
# df['gen_sql'] = gen_sqls
#
# # 평가를 위한 requests.jsonl 생성
# eval_filepath = "text2sql_evaluation.jsonl"
# make_requests_for_gpt_evaluation(df, eval_filepath)

print(hf_pipe(example, do_sample=False, return_full_text=False, max_length=512, truncation=True))