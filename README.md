# checKHUmate
KHUDA 5기 심화 프로젝트 룸메이트 추천 알고리즘 개발

## Table of Contents
* [Technologies](#technologies)
* [Result](#Result)
* [Members](#Members)


## Key Features
Our project has two key features: `feature similarity` and `sentence similarity`.

## Service Architecture
![checKHUmate_arc](https://github.com/ChecKHUMate/Demo_ChecKHUMate/assets/122276734/66944f50-c6fa-4d2d-a8ce-d9de28693107)

## Data Collection
- Collected data from 2021 to 2024 from everytime's dormitory roommate checklist image data, resulting in 133 rows of data.
- Separated the collected data into user data with 10 attributes and wish data with 9 attributes for database entry.
    - The attributes collected are age, student ID, gender, major, bedtime, cleaning frequency, smoking status, frequency of drinking, and MBTI.
- Specifically, for the user data, a placeholder for a one-line introduction is included. This is to recommend people with similar keywords based on the one-line introduction.


## feature similarity

- Calculate the feature similarity between each user's characteristics and the desired characteristics of the wish_user (gender is set to be the same).
- Generate the Top 10 matches by index(user_id) based on the most similar features.

![faiss](https://github.com/ChecKHUMate/Demo_ChecKHUMate/assets/122276734/c445b85f-2728-4710-8b01-6c2e39aa5f8d)



## sentence embedding

- Input the pairs of sentences corresponding to the top 10 matches obtained from feature similarity.
- Convert each input sequence into embedding vectors using a pre-trained BERT model.
- Perform a pooling operation (typically mean-pooling) on the converted embedding vectors to transform them into sentence embedding vectors.

![SBERT](https://github.com/ChecKHUMate/Demo_ChecKHUMate/assets/122276734/7daf16a4-f8de-48e1-b68b-fd6f80dc0f63)

## sentence similarity

- Extract index similarity between embedding vectors for each user_id obtained through sentence embedding.
- For each row (user_id), calculate the similarity between the one_sentence of the user_id corresponding to the row number and the sentence embedding vectors of the other 10 indices.
- Return the top 2 matches based on the highest similarity.

![Result](https://github.com/ChecKHUMate/Demo_ChecKHUMate/assets/122276734/380c2f4c-f12d-4ec7-95cd-9166691ea9e7)


## Project Archive
This is our notion page for our project archive. : 
[Notion](https://baram1ng.notion.site/KHUDA-RecSys-5ea8676b7294402e81dc92cce990556d?pvs=4)

## Members
|멤버이름|역할|
|------|---|
|배아람|Leader, Data Preprocessing, Data Modeling, Frontend|
|이준혁|Data Collection, Data Modeling|
|이다은|Data Collection, Frontend, Backend|
|이수민|Data Collection, Design, Frontend|
|조준영|Data Collection, Design, Frontend|
|한주상|Data Collection, Frontend|
