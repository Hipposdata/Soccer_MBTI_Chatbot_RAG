# Soccer_MBTI_Chatbot_RAG
축구 MBTI 챗봇 With RAG

## 서비스 개발 동기

Q. 축구인으로서 나의 축구 플레이 스타일이나 성향이 어떻게 평가될까?  
Q. 나와 비슷한 플레이 스타일을 가진 선수는 누가 있을까?  

-> 축구 MBTI를 통해 나의 축구 플레이 스타일과 나와 비슷한 스타일의 선수를 추천해주는 챗봇 서비스 제작


## 프로젝트 수행 방법

### 축구 MBTI지표 정의 및 유사 단어 정의
![축구MBTI지표그림](https://github.com/user-attachments/assets/f766f42b-d888-45af-8c1e-e50c0bb7f70f)  
4가지 항목, 총 16유형  

1. 공격형(A, Attack) vs 수비형(D, Defense)
2. 협동형(T, Teamwork) vs 단독형(P, Personal)
3. 핵심형(K, Key) vs 조커형(J, Joker)
4. 적극형(E, Extroversion) vs 소극형(I, Introversion)
 
1. 공격형(A) / 수비형(D) 지표: 유효슈팅 수 + 슈팅 수 + 리커버리 수  

공격형: 공격적, 진격, 기습, 침략, 도전, 도전적, 공격적인, 진격하는 등등  
수비형: 방호적, 보호적, 방어적,막기, 방어적인, 보호적인 등등  

2. 협동형(T ) / 단독형(P) 지표: 패스성공 수  

협동형: 협동적, 팀워크 지향, 집단적, 공동체 중심, 협력적, 팀끼리, 조직적 등등  
단독형: 주관적, 개별적, 자기만의, 고유한, 스스로 등등  

3. 핵심형(K) / 조커형(J) 지표: 출전시간(분) ÷ 총 경기수    
 
핵심형: 필수적인, 오랜 출전시간, 주축 선수, 에이스, 중심적인, 신뢰받는, 핵심 전력 등등  
조커형: 조커, 슈퍼 서브, 임팩트 플레이어, 교체 카드, 후반전 강자, 분위기 반전, 한 방이 있는 선수, 기습적인, 결정적 순간의 카드, 경기 흐름 전환자 등등  

4. 적극형(E) / 소극형(I) 지표: 경고 수 ÷ 총 경기수  
적극형: 공격적, 거친 플레이, 피지컬, 태클, 몸싸움, 압박, 강한 수비, 저돌적, 과감한 등등  
소극형: 신중한, 조용한 플레이, 클린 플레이, 조심스러운, 소극적 수비, 자제력, 안전한 플레이, 침착한 등등


### 2023 ~ 2024 시즌 EPL 선수 기록 데이터 크롤링 - 네이버 스포츠 링크
https://m.sports.naver.com/wfootball/record/epl?seasonCode=QFmj&tab=players

선수 기록을 기반으로 해당 선수의 축구 MBTI산출 
각 항목에 해당하는 축구 지표의 모든 선수의 평균값을 기준으로 평균 이상, 이하로 구분하여 각 축구 MBTI 항목 구분

### RAG(Retrieval-Augmented Generation) 적용
정의한 축구 MBTI지표 문서 + 2023 ~ 2024 시즌 EPL 선수 기록 데이터 문서 기반으로 RAG적용
질문과 각 지표들간의 유사도를 계산하여 높은 유사도 값을 해당 MBTI 유형으로 판단

### RAG 사용 유무 결과 비교   
기본 gpt모델 (gpt-4o-mini)   
![image](https://github.com/user-attachments/assets/6b3a0a79-ddbb-4a47-8481-a20520f215d1)

 
기본 gpt모델 (gpt-4o-mini) + RAG  
![image](https://github.com/user-attachments/assets/631563ae-99b3-4184-b5d9-254cfbc863c7)
![image](https://github.com/user-attachments/assets/e0ba34c4-5e74-4f1d-8124-db0628e3359c)

<figure class="half">
<figure class="half">  
 <a href="link"><img src="![image](https://github.com/user-attachments/assets/631563ae-99b3-4184-b5d9-254cfbc863c7)"></a>  
 <a href="link"><img src="이미지경로"></a> 
</figure>

-> 정의한 MBTI 문서 기반 답변 및 최근 선수(2023~2024 시즌) 추천 

## STEAMLIT DEMO  
https://github.com/user-attachments/assets/3967f835-976f-483c-8f3f-a08eeef9fc3a  

## How to run  
<br>

```bash
pip install -r requirements.txt
```

```bash
cd streamlit
streamlit run mbti_web.py
```

# Reference
한동대학교 축구 빅데이터 캠프 - 나는 데이터로 축구한다(LLM과 RAG 기술 활용편)
[https://github.com/Hipposdata/facamp-2025-winter](https://github.com/ai-5050/facamp-2025-winter)
