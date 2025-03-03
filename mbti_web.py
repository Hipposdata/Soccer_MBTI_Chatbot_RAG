#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import json
import shutil
import os
import time
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# **Streamlit UI 설정**
st.title("⚽️ 축구 MBTI 챗봇 With RAG")

# **1️⃣ 사용자 입력받기**
query = st.text_area("💬 질문을 입력하세요:")

if st.button("🔎 분석 시작"):

    # **OpenAI 모델 초기화**
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # **2️⃣ JSON 파일 로드 (축구 MBTI 데이터)**
    file_path = "./news-sample1.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # **3️⃣ JSON에서 MBTI 관련 키워드 자동 추출**
    mbti_keywords = set()
    for entry in data:
        page_content = entry.get("page_content", "")
        words = page_content.split()
        mbti_keywords.update(words)

    # **4️⃣ 사용자의 질문이 MBTI 관련인지 판별**
    is_mbti_question = any(word in query for word in mbti_keywords)

    # **5️⃣ 챗봇의 `persona` 설정 (축구 MBTI 전문가지만 다른 질문도 가능)**
    system_message = SystemMessage(
        content="""당신은 축구 MBTI 전문가입니다.  
        사용자가 축구와 관련된 질문을 하면 MBTI 기반으로 분석하여 답변합니다.  
        그러나 사용자가 다른 주제에 대한 질문을 하면, 가능한 한 신뢰할 수 있는 답변을 제공합니다.  
        """
    )

    if is_mbti_question:
        st.write("📌 **축구 MBTI 관련 질문으로 인식되었습니다. 분석을 시작합니다...**")

        # **6️⃣ MBTI 분석 진행**
        title_to_index = {
            entry["metadata"]["title"]: idx
            for idx, entry in enumerate(data)
            if "metadata" in entry and "title" in entry["metadata"]
        }

        documents = [Document(page_content=entry["page_content"], metadata=entry["metadata"]) for entry in data]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        DB_PATH = "./chroma_db"
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH, ignore_errors=True)
        os.makedirs(DB_PATH, exist_ok=True)

        start_time = time.time()

        db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=DB_PATH
        )

        end_time = time.time()
        st.write(f"🔹 Chroma DB 생성 완료 (소요 시간: {end_time - start_time:.2f}초)")

        # **7️⃣ MBTI 분석 (TF-IDF 유사도 비교)**
        pairs = [("A", "D"), ("P", "T"), ("K", "J"), ("E", "I")]

        vectorizer = TfidfVectorizer()
        page_contents = [doc.page_content for doc in docs] + [query]  
        tfidf_matrix = vectorizer.fit_transform(page_contents)
        query_vector = tfidf_matrix[-1]

        selected_titles = []

        for title1, title2 in pairs:
            if title1 not in title_to_index or title2 not in title_to_index:
                continue
            idx1 = title_to_index[title1]
            idx2 = title_to_index[title2]
            similarity1 = cosine_similarity(query_vector, tfidf_matrix[idx1])[0][0]
            similarity2 = cosine_similarity(query_vector, tfidf_matrix[idx2])[0][0]
            chosen_title = title1 if similarity1 > similarity2 else title2
            selected_titles.append(chosen_title)

        mbti_result = "".join(selected_titles)
        st.write(f"📌 MBTI 결과: **{mbti_result}**")

        # **🏆 선수 추천**
        processed_file_path = './player.xlsx'
        processed_data = pd.read_excel(processed_file_path)
        filtered_data = processed_data[processed_data['MBTI 코드'] == mbti_result]

        if filtered_data.empty:
            st.warning("⚠️ 해당 MBTI 코드와 일치하는 선수가 없습니다.")
        else:
            filtered_data['경기'] = pd.to_numeric(filtered_data['경기'], errors='coerce')
            top_3_players = filtered_data.nlargest(3, '경기')[['선수', '팀', '경기']]
            top_3_players_str = top_3_players.to_string(index=False)
            st.write("🏆 추천 선수:")
            st.dataframe(top_3_players)

            # **🔟 MBTI 질문일 때 특정 템플릿을 사용하여 답변**
            chat_template1 = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=f"""당신은 축구 전문가이면서 심리행동전문가야.
                        사용자가 너에게 말하는 성향을 바탕으로 우리가 만든 축구 MBTI를 분석하고, 사용자에게 그 결과를 알려줘.
                        A(attack) 
                        D(defence) 
                        T(team)
                        P(personal)
                        K(key) 
                        J(joker)
                        E(extroversion)
                        I(Introverted)


                        이 순서대로 해당되는 영어에 따라 해당되는 4개를 설명해줘
                        
                        꼭 축구MBTI {mbti_result} 사용자에게 4가지 유형으로 제공해야 하고,  
                        정보가 부족해도 일단 사용자가 제공한 데이터로 알려줘!
                        결과에 대한 설명을 3줄 이상으로 길고 자세하게 풀어서 설명해줘.
                        마지막에는 사용자의 MBTI {mbti_result} 결과와 같은 축구 선수 3명을 추천해줘.

                        아래는 2023~2024 시즌 EPL리그에서 뛰었던 비슷한 유형의 선수 데이터야:
                        {top_3_players_str}
                        밑에 각 선수에 대한 소속팀, 포지션, 플레이스타일, 나이, 국적, 평가 등등 해당 선수에 대한 정보를 매우 구체적으로 알려줘
                        <예시>
                        ***2023~2024 시즌 EPL리그에서 뛰었던 비슷한 유형의 선수를 추천하겠습니다***
                        소속팀:
                        나이:
                        국적:
                        포지션:
                        평가:
                        """
                    ),
                    HumanMessage(content=query)
                ]
            )

            messages1 = chat_template1.format_messages(text=query)
            st.write("📊 **MBTI 분석 결과**")

            # 스트리밍 출력을 위한 placeholder 생성
            output_placeholder = st.empty()

            output = ""
            for chunk in model.stream(messages1):
                output += chunk.content
                output_placeholder.write(output)

    else:
        # **MBTI 관련이 아닌 질문이면 OpenAI를 이용해 일반 응답 제공**
        st.write("💡 일반 질문으로 인식되었습니다. OpenAI가 답변을 생성 중...")
        messages = [
            system_message,  # ✅ 설정된 역할을 반영
            HumanMessage(content=query)
        ]
        response = model.invoke(messages)
        st.write("📖 **OpenAI 응답:**")
        st.write(response.content)
