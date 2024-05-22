import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import faiss
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings('ignore')


def prepare_data(data):
    """
    주어진 데이터 프레임에서 필요한 정보를 추출하고 포맷팅하는 함수.

    Parameters:
        data (pd.DataFrame): 원본 데이터 프레임.

    Returns:
        user_data (pd.DataFrame): 사용자 데이터 프레임.
        wish_data (pd.DataFrame): 사용자의 선호에 관한 데이터 프레임.
    """
    # 사용자 데이터 선택
    user_data_columns = [
        'user_id', 'domitory', 'age', 'student_id', 'gender', 'major',
        'bedtime', 'clean_duration', 'smoke', 'alcohol', 'mbti', 'one_sentence'
    ]
    user_data = data[user_data_columns]
    user_data = user_data.set_index('user_id')
    
    # 사용자 선호 데이터 선택
    wish_data_columns = [
        'user_id', 'wish_domitory', 'wish_age', 'wish_student_id', 'wish_gender',
        'wish_major', 'wish_bedtime', 'wish_clean_duration', 'wish_smoke',
        'wish_alcohol', 'wish_mbti'
    ]
    wish_data = data[wish_data_columns]
    wish_data = wish_data.set_index('user_id')
    
    return user_data, wish_data
  
  # 코랩용.
def encode_and_concat_to_array(dataframe, column_name):
    """
    원핫 인코딩을 수행하고 원본 데이터프레임에 결과를 추가한 후 전체를 numpy 배열로 반환합니다.

    Args:
    dataframe (pd.DataFrame): 원본 데이터를 포함한 데이터프레임.
    column_name (str): 원핫 인코딩을 적용할 열의 이름.

    Returns:
    numpy.ndarray: 원핫 인코딩된 결과와 원본 데이터가 결합된 numpy 배열.
    """
    # 원핫 인코더 초기화 및 피팅
    encoder = OneHotEncoder(sparse_output=False)  # 바로 numpy array 반환 설정
    encoded_data = encoder.fit_transform(dataframe[[column_name]])

    # 원핫 인코딩 결과를 DataFrame으로 변환
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column_name]))

    # 데이터프레임의 인덱스 리셋
    dataframe = dataframe.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)

    # 원본 데이터와 원핫 인코딩된 데이터를 결합
    new_dataframe = pd.concat([dataframe, encoded_df], axis=1)

    # 인코딩된 열 제거
    new_dataframe = new_dataframe.drop(column_name, axis=1)

    return new_dataframe

# 각 사용자의 wish_data를 기반으로 user_data에서 자기 자신을 제외한 상위 k개의 가장 유사한 사용자들을 찾아낸 결과를 배열로 반환합니다.
class Recommender_without_sentence:
    def __init__(self, user_data, wish_data):
        self.user_data = user_data
        self.wish_data = wish_data
        self.index = None
        self.d = user_data.shape[1]

    def build_index(self):
        """FAISS 인덱스를 생성하고 유저 데이터를 추가합니다."""
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(self.user_data.astype('float32'))

    def find_roommates(self, k=10):
        """각 유저의 wish_data에 대해 가장 유사한 유저를 찾되, 자기 자신은 제외하고, -1 인덱스가 발생한 경우도 기록합니다."""
        if self.index is None:
            self.build_index()

        all_results = []  # 결과를 저장할 배열
        invalid_indices_info = []  # -1 인덱스가 발생한 정보를 저장할 배열

        for i in range(self.wish_data.shape[0]):
            # 자기 자신을 포함하여 k+1개의 결과를 검색
            distances, indices = self.index.search(self.wish_data[i:i+1].astype('float32'), k+1)

            # 자기 자신의 인덱스와 -1을 제외
            valid_indices = indices[0][(indices[0] != i) & (indices[0] != -1)]

            # -1이 발생한 경우 기록
            if np.any(indices[0] == -1):
                invalid_indices_info.append((i, list(indices[0])))

            # k개의 결과만 반환
            all_results.append(valid_indices[:k])

        # 결과와 -1 정보 모두 반환
        return np.array(all_results)
      
# 신버전
class IntroductionRecommender:
    def __init__(self, user_data):
        self.user_data = user_data
        self.embedder = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.index = None
        self.id_index = np.array(self.user_data.index)
        self.embeddings = None
        self.user_data['one_sentence'] = user_data['one_sentence'].fillna('No introduction provided').astype(str)

    def create_faiss_index(self):

        """임베딩을 생성하고 FAISS 인덱스에 추가합니다."""
        # 모든 자기소개 문장 임베딩 생성
        self.embeddings = self.embedder.encode(self.user_data['one_sentence'].tolist(), convert_to_tensor=False)

        normalized_embeddings = self.embeddings.copy()

        # normalized_embeddings의 벡터를 L2 Norm(유클리드 거리 기준)으로 정규화합니다.
        # 이렇게 함으로써, 벡터의 길이를 1로 만들어 줍니다. 이것은 내적 기반의 유사도 검색에서 중요한 절차입니다.
        # 왜냐하면 정규화된 벡터들 사이의 내적은 두 벡터의 코사인 유사도와 동일하기 때문입니다.
        faiss.normalize_L2(normalized_embeddings)

        # FAISS 인덱스 생성
        d = self.embeddings.shape[1]  # 임베딩 차원
        index_flat = faiss.IndexFlatIP(d)

        self.index =index = faiss.IndexIDMap(index_flat)
        self.index.add_with_ids(normalized_embeddings, self.id_index)


    def find_similar_introductions(self, result_indices, k=10):
        if self.index is None:
            raise ValueError("FAISS index is not built. Please run create_faiss_index method first.")

        all_group_similarities = []

        for indices in result_indices:
            min_distances = {}  # 인덱스별 최소 거리를 저장할 사전

            for element in indices:
                # 인덱스에 해당하는 텍스트만 인코딩합니다.
                text_to_encode = [self.user_data['one_sentence'].iloc[element]]
                vector = self.embedder.encode(text_to_encode)
                faiss.normalize_L2(vector)

                D, I = self.index.search(vector.reshape(1, -1), k+1)

                for i, d in zip(I[0], D[0]):
                    if i != element and i in indices:
                        if i not in min_distances or min_distances[i] > d:
                            min_distances[i] = d  # 최소 거리 업데이트


            # 인덱스와 거리를 튜플로 변환하여 리스트로 만든 후, 텍스트 유사도에 따라
            group_similarities = sorted(min_distances.items(), key=lambda x: x[1], reverse = True)
            all_group_similarities.append(group_similarities)

        return all_group_similarities