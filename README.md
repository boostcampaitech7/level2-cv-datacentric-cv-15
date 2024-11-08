# 🧾 다국어 영수증 OCR

<br/>

## 👨‍👩‍👧‍👦 팀 구성
<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/SeoJinHyoung">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003813%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>서진형</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/andantecode">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003899%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>함로운</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sihari-1115">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004046%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>이시하</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/IronNote">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004085%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김명철</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ruka030809">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004086%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김형준</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/alexminyoungpark">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00004104%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>박민영</b></sub><br />
      </a>
    </td>
  </tr>
</table>
</div>
<br />

## 📃 프로젝트 개요
OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR은 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

- 본 대회에서는 다국어 (중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에 대한 OCR task를 수행합니다.

- 본 대회에서는 글자 검출만을 수행합니다. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작합니다.

- 본 대회는 제출된 예측 (prediction) 파일로 평가합니다.

- 대회 기간과 task 난이도를 고려하여 코드 작성에 제약사항이 있습니다. 

`본 대회는 Data-Centric AI의 관점에서 모델 활용을 경쟁하는 대회입니다. 이에 따라 제공되는 베이스라인 코드 중 모델 관련 부분을 변경하는 것이 금지되어 있습니다.`

<br/>

## ✔ 평가 지표
`DetEval 방식으로 평가됩니다.`

- 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가방법

- recall과 precision 의 조화평균인 F1 score 를 기준으로 랭킹이 산정

<br/>

## 📅 프로젝트 일정
2024/10/30 ~ 2024/11/07
<br/>

## 💻 개발 환경
```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
  - GPU : Tesla V100-SXM2 32GB × 4
- Framework : PyTorch, numba, augraphy
- Collaborative Tool : Git, Notion
```
<br/>

## 📁 프로젝트 구조
```
📦level2-cv-datacentric-cv-15
 ┣ 📂.github
 ┃ ┗ 📄.keep
 ┣ 📂eda
 ┃ ┣ 📄eda.ipynb
 ┣ 📂code
 ┃ ┣ 📄dataset.py
 ┃ ┣ 📄ensemble.py
 ┃ ┣ 📄inference.py
 ┃ ┣ 📄train.py
 ┃ ┣ 📄transform.py
 ┃ ┣ 📄README.md
 ┣ 📂utils
 ┃ ┣ 📄mislabel_fix.ipynb
 ┃ ┣ 📄visualize.ipynb
 ┣ 📄.gitignore
 ┣ 📄README.md
 ```
<br/>
 
#### 1) `eda` 
- 다국어 영수증 이미지 데이터셋 분석 노트북

#### 2) `code` 
- EAST 모델 입력 전 전처리 및 데이터 증강 관련 코드
- 학습 및 추론 코드

#### 3) `utils`
- 데이터 클렌징 및 모델 추론 및 데이터셋 시각화 코드

<br/>

## 🔆 프로젝트 결과
- public

![image](https://github.com/user-attachments/assets/06f6f080-f37e-414e-aaeb-7dc67ad21992)

- private

![image](https://github.com/user-attachments/assets/3c55aa9e-beb5-4f5d-a36b-1f7d8fb391b9)
<br/>


## 📃 보고서

- [랩업 리포트]()
- [프레젠테이션](https://drive.google.com/file/d/1UNRjtY4LtVj45dqpMo-mJXRnIkWtKV02/view?usp=sharing)

<br/>
<br/>
<br/>



    - 실험 내용 및 상세 결과는 보고서에 기술