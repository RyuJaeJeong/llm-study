from datasets import load_dataset


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


df_sql = load_dataset("shangrilar/ko_text2sql", "origin")['train']
df_sql = df_sql.to_pandas()
df_sql = df_sql.dropna().sample(frac=1, random_state=42)
df_sql = df_sql.query("db_id != 1")

for idx, row in df_sql.iterrows():
    df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'])

df_sql.to_csv('data/train.csv', index=False)