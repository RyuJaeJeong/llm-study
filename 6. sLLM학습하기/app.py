from pathlib import Path
import pandas as pd
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
# pip install transformers==4.40.1 bitsandbytes==0.43.1 accelerate==0.29.3 datasets==2.19.0 tiktoken==0.6.0
# pip install huggingface_hub==0.22.2 autotrain-advanced==0.7.77

# 프롬프트 생성
def make_prompt(ddl, question, query=''):
  prompt = f"""당신은 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결 할 수 있는 SQL 쿼리를 생성하세요.
  ###DDL:
  {ddl}
  ###Question:
  {question}
  ### SQL:
  {query}"""
  return prompt

"""
========================================
평가 파이프라인 start
========================================
"""
# gpt 평가를 위한 요청 json파일
def make_requests_for_gpt_evaluation(df, filename, dir='requests'):
  if not Path(dir).exists():
    Path(dir).mkdir(parents=True)
  prompts = []
  for idx, row in df.iterrows():
    prompts.append("""
    Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else return "no".
    Output JSON Format: {"resolve_yn":""}""" + f"""
      DDL: {row['context']}
      Question: {row['question']}
      gt_sql: {row['answer']}
      gen_sql: {row['gen_sql']}
    """)
  jobs = [{"model": "gpt-4-turbo-preview", "response_format" : {"type": "json_object"}, "messages": [{"role":"system", "content": prompt}]} for prompt in prompts]
  with open(Path(dir, filename), "w") as f:
    for job in jobs:
      json_string = json.dumps(job)
      f.write(json_string+ "\n")


# 결과 jsonl 파일을 csv로 변환하는 함수
def change_jsonl_to_csv(input_file, output_file, prompt_column="prompt", response_column="response"):
  prompts = []
  responses = []
  with open(input_file, 'r') as json_file:
    for data in json_file:
      prompts.append(json.loads(data)[0]['messages'][0]['content'])
      responses.append(json.loads(data)[1]['choices'][0]['message']['content'])
  df = pd.DataFrame({prompt_column: prompts, response_column: responses})
  df.to_csv(output_file, index=False)
  return df

"""
========================================
평가 파이프라인 end
========================================
"""

def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",            # 자동으로 GPU 할당
        load_in_4_bit=True,
        bnb_4bit_compute_dtype=torch.float16        
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


model_id = "models/Yi-Ko-6B"
hf_pipe = make_inference_pipeline(model_id)
example = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL
CREATE TABLE players (
  player_id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  date_joined DATETIME NOT NULL,
  last_login DATETIME
);

### Question:
사용자 이름에 'admin'이 포함되어있는 계정의 수를 알려주세요

###SQL:
"""
result = hf_pipe(example, do_sample=False, return_full_text=False, max_length=512, truncation=True)
print(result)

# df = load_dataset("shangrilar/ko_text2sql", "origin")["test"]
# df = df.to_pandas()
# for idx, row in df.iterrows():
#   prompt = make_prompt(row['context'], row['question'])
#   df.loc(idx, 'prompt') = prompt

# # sql 생성
# gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False, return_full_text=False, max_length=1024, truncation=True)
# gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
# df['gen_sql'] = gen_sqls

# #평가를 위한 requests.jsonl 생성
# eval_filepath = "text2sql_evaluation.jsonl"
# make_requests_for_gpt_evaluation(df, eval_filepath)