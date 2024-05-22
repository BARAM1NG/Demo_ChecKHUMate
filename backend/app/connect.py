import pandas as pd
import mysql.connector

# CSV 파일 읽기
df = pd.read_csv('/mnt/data/user_one_sentence - 시트1.csv', encoding='utf-8')

# MySQL 데이터베이스에 연결
conn = mysql.connector.connect(
    host='localhost',
    user='ChecKHUMate',
    password= None,
    database='your_database',
    charset='utf8mb4'
)

# 데이터베이스 커서 생성
cursor = conn.cursor()

# 데이터베이스에 데이터 삽입
for sentence in df['one_sentence']:
    query = "INSERT INTO sentences (one_sentence) VALUES (%s)"
    cursor.execute(query, (sentence,))

# 변경 사항 커밋
conn.commit()

# 연결 종료
cursor.close()
conn.close()
