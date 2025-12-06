[Readme.txt](https://github.com/user-attachments/files/23980422/Readme.txt)
---

# 🏷️ Q-COST AI APP (Capstone Project)

### **Streamlit 기반 중소기업 품질비용(Q-COST) 분석 & 의사결정 지원 시스템**

본 프로젝트는 캡스톤 디자인 과제로 개발된 **Streamlit 기반 Q-COST AI Assistant**입니다.
중소기업의 품질비용 구조(Q-COST: *Prevention · Appraisal · Internal Failure · External Failure*)를 자동 분석하고,
**AI 기반 품질 리스크 예측·비용 절감 의사결정**을 지원합니다.

---

## 🚀 프로젝트 개요

품질관리(Quality Management)에서 Q-COST는
**예방비용(Prevention)** → **평가비용(Appraisal)** → **내부 실패비용(Internal Failure)** → **외부 실패비용(External Failure)**
로 구성되며 기업의 품질 경쟁력을 결정하는 핵심 지표입니다.

그러나 실제 중소기업 환경에서는:

* 품질 데이터가 비정형적이고
* 비용 구조 분석이 어렵고
* 개선 활동이 실제 실패비용 절감으로 이어졌는지 판단하기 어렵습니다.

본 프로젝트는 이러한 문제를 해결하기 위해
**AI 기반 자동 분석 + 시뮬레이션 + 대화형 컨설팅 기능**을 갖춘 시스템을 구현했습니다.

---

## 🎯 주요 목표

* Q-COST 비용 구조 자동 파악
* **RandomForest / XGBoost / Logistic·Linear Regression** 기반 품질예측
* **SHAP DataFrame 중요도 분석** 제공
* 예방·평가비용 조정 시 실패비용 변화 시뮬레이션 제공
* **Gemini API 기반 Q-COST 컨설턴트 제공**

---

## ✨ 핵심 기능

### 1️⃣ 파일 업로드 기반 자동 분석

* CSV/XLSX 데이터 업로드
* Q-COST 비용(예방·평가·실패) 자동 탐지
* 모델링용 타깃(분류/회귀) 자동 인식
* 사용자 커스텀 비용 매핑 가능

---

### 2️⃣ AI 모델링 & 성능 분석

#### 지원 모델

* Logistic Regression / Linear Regression
* **RandomForest Classifier & Regressor**
* **XGBoost Classifier & Regressor** (옵션)

#### 제공 분석

* SHAP 기반 변수 중요도(DataFrame)
* 주요 성능 지표:

  * Classification: **Accuracy / F1 / AUC**
  * Regression: **RMSE / R²**

---

### 3️⃣ Q-COST 시나리오 분석

* 예방비용 / 평가비용을 **±50 ~ 200%** 조정
* 조정된 비용 시나리오에 따른 **내부·외부 실패비용 변화 예측**
* RandomForest 회귀 기반 시뮬레이션 엔진

---

## 🛠 기술 스택

| 분야                | 기술                           |
| ----------------- | ---------------------------- |
| **Frontend**      | Streamlit                    |
| **Modeling**      | Scikit-learn, XGBoost, SHAP  |
| **Backend Logic** | Python, Pandas, NumPy        |
| **AI Assistant**  | Google Gemini API            |
| **Infra**         | GitHub |

----
