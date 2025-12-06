import os
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import shap
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# -----------------------
# 유틸 함수들
# -----------------------
def read_csv_auto(file_obj):
    """
    업로드된 CSV 파일을 인코딩을 바꿔 가며 읽는 함수.
    UTF-8이 안 되면 cp949, 그래도 안 되면 ISO-8859-1 시도.
    """
    import pandas as pd

    # 1) UTF-8 먼저 시도
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="utf-8")
    except UnicodeDecodeError:
        pass

    # 2) cp949 (한글 윈도우 기본)
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="cp949")
    except UnicodeDecodeError:
        pass

    # 3) 마지막으로 ISO-8859-1 같은 범용 인코딩
    file_obj.seek(0)
    return pd.read_csv(file_obj, encoding="iso-8859-1", errors="replace")


def detect_cost_columns(df):
    """
    Q-COST 관련 컬럼 자동 탐지 (추천용)
    - 예방비용(prevention)
    - 평가/검사비용(appraisal)
    - 내부 실패(internal failure)
    - 외부 실패(external failure)
    """
    cols = df.columns

    def match_keywords(keywords):
        return [
            c for c in cols
            if any(k.lower() in str(c).lower() for k in keywords)
        ]

    prevention_cols = match_keywords(["예방", "prevention", "prevention_cost", "prevention cost"])
    appraisal_cols = match_keywords(["평가", "검사", "inspection", "appraisal"])
    internal_failure_cols = match_keywords(["내부", "internal_failure", "internal failure"])
    external_failure_cols = match_keywords(["외부", "external_failure", "external failure"])

    return {
        "prevention": prevention_cols,
        "appraisal": appraisal_cols,
        "internal_failure": internal_failure_cols,
        "external_failure": external_failure_cols,
    }


def detect_target_column(df):
    """
    성공/실패, 양품/불량 등 타깃 컬럼 자동 탐지
    - 값이 2~3개 정도인 컬럼 + 이름 패턴 기반
    """
    candidates = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if 1 < len(unique_vals) <= 3:
            candidates.append(col)

    # 이름 패턴 우선 탐색
    preferred_patterns = [
        "성공", "실패", "합격", "불합격", "불량", "양품",
        "pass_fail", "passfail", "pass", "fail",
        "target", "label", "y"
    ]
    for col in candidates:
        name = str(col).lower()
        if any(p.lower() in name for p in preferred_patterns):
            return col

    # 그래도 없으면 첫 번째 후보 또는 None
    return candidates[0] if candidates else None


def binarize_target(series):
    """
    성공/실패 텍스트를 0/1로 맵핑
    """
    s = series.copy()

    mapping = {
        "성공": 1, "success": 1, "pass": 1, "합격": 1,
        "실패": 0, "fail": 0, "불합격": 0, "불량": 0, "양품": 1
    }

    def _map(v):
        if pd.isna(v):
            return np.nan
        v_str = str(v).strip().lower()
        if v_str in mapping:
            return mapping[v_str]
        # 숫자는 그대로
        try:
            return float(v)
        except Exception:
            return np.nan

    s = s.map(_map)

    # 0/1만 남기기
    unique_vals = [u for u in s.dropna().unique()]
    # 이미 0/1이면 그대로
    if set(unique_vals).issubset({0, 1}):
        return s

    # 2개 값이면 작은 값 0, 큰 값 1로 강제 매핑
    if len(unique_vals) == 2:
        lo, hi = sorted(unique_vals)
        return s.map(lambda x: 0 if x == lo else (1 if x == hi else np.nan))

    # 그 외에는 그대로 반환 (나중에 연속형으로 취급)
    return s


def train_models_classification(X, y):
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1) 로지스틱 회귀
    logreg = LogisticRegression(max_iter=500)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]

    results["Logistic Regression"] = {
        "model": logreg,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    # 2) 랜덤포레스트
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    results["RandomForest"] = {
        "model": rf,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    # 3) XGBoost (있으면)
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        y_prob = xgb_model.predict_proba(X_test)[:, 1]

        results["XGBoost"] = {
            "model": xgb_model,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
        }

    return results, (X_train, X_test, y_train, y_test)


def train_models_regression(X, y):
    """
    회귀 모델 학습 (squared=False 안 쓰고, RMSE 직접 계산)
    """
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1) 선형 회귀
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results["Linear Regression"] = {
        "model": lr,
        "rmse": rmse,
        "r2": r2_score(y_test, y_pred),
    }

    # 2) 랜덤포레스트
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results["RandomForest"] = {
        "model": rf,
        "rmse": rmse,
        "r2": r2_score(y_test, y_pred),
    }

    # 3) XGBoost (있으면)
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        results["XGBoost"] = {
            "model": xgb_model,
            "rmse": rmse,
            "r2": r2_score(y_test, y_pred),
        }

    return results, (X_train, X_test, y_train, y_test)

def remove_failure_related_features(X, failure_cols):
    """
    실패비용 관련 컬럼(내부/외부/통합)을 X(입력 특징)에서 제거하여
    모델 누출(leakage)을 방지한다.
    """
    failure_cols = [c for c in failure_cols if c in X.columns]
    return X.drop(columns=failure_cols, errors="ignore")


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    # (그래프용 함수 – 지금은 호출하지 않지만 남겨둠)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        st.info("이 모델은 feature_importances_ 속성이 없습니다.")
        return

    idx = np.argsort(importances)[::-1]
    sorted_names = np.array(feature_names)[idx]
    sorted_vals = importances[idx]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(sorted_vals)), sorted_vals)
    plt.xticks(range(len(sorted_vals)), sorted_names, rotation=90)
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()


def plot_shap_summary_tree(model, X_train, feature_names, title="SHAP Summary"):
    # (그래프용 함수 – 지금은 호출하지 않지만 남겨둠)
    st.write(f"### {title}")
    sample = X_train
    if X_train.shape[0] > 500:
        sample = X_train.sample(500, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    try:
        # 이진 분류인 경우 shap_values[1] 사용
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap.summary_plot(shap_values[1], sample, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
        st.pyplot(plt.gcf())
    finally:
        plt.close()


def build_scenario_result(df, cost_cols, failure_col_name=None):
    """
    예방/평가 비용을 변경했을 때 실패비용 변화 시뮬레이션
    - 실패비용: 사용자가 지정한 통합 실패비용 컬럼 또는 내부/외부 실패비용 합계
    - 타깃: 실패비용 (회귀)
    """
    # 예방/평가 비용 컬럼
    prevention_cols = cost_cols.get("prevention", [])
    appraisal_cols = cost_cols.get("appraisal", [])
    internal_failure_cols = cost_cols.get("internal_failure", [])
    external_failure_cols = cost_cols.get("external_failure", [])

    # 숫자형 데이터만 사용
    df_num = df.select_dtypes(include=[np.number]).copy()

    # 1) 우선 사용자가 지정한 통합 실패비용 컬럼을 사용
    failure_cols = []
    if failure_col_name is not None and failure_col_name in df_num.columns:
        failure_cols = [failure_col_name]
    else:
        # 2) 통합 실패비용 컬럼이 없으면 내부/외부 실패비용 합산
        failure_cols = [c for c in (internal_failure_cols + external_failure_cols) if c in df_num.columns]

    if not failure_cols:
        return None, "실패비용(타깃) 수치 컬럼을 찾을 수 없습니다. 통합 실패비용 컬럼을 지정하거나 내부/외부 실패비용 컬럼을 선택해 주세요."

    # 실패비용 타깃 생성
    df_num["failure_cost"] = df_num[failure_cols].sum(axis=1)

    # 특징 변수로 사용할 후보 (예방 + 평가 비용 포함)
    feature_cols = list(set(
        [c for c in prevention_cols + appraisal_cols if c in df_num.columns]
    ))

    # 없으면 전체 숫자 컬럼에서 failure_cost 제외하고 사용
    if not feature_cols:
        feature_cols = [c for c in df_num.columns if c != "failure_cost"]

    # 결측치 제거
    data = df_num[feature_cols + ["failure_cost"]].dropna()
    if data.shape[0] < 20:
        return None, "시뮬레이션을 하기에는 유효한 데이터가 너무 적습니다."

    X = data[feature_cols]
    y = data["failure_cost"]

    failure_related_cols = internal_failure_cols + external_failure_cols
    if failure_col_name is not None:
        failure_related_cols.append(failure_col_name)

    X = remove_failure_related_features(X, failure_related_cols)

    results, (X_train, X_test, y_train, y_test) = train_models_regression(X, y)

    # 가장 단순한 랜덤포레스트 사용
    if "RandomForest" in results:
        model = results["RandomForest"]["model"]
    else:
        model = list(results.values())[0]["model"]

    # baseline: 전체 평균값 한 점에서 예측
    base_point = X.mean(axis=0).to_frame().T

    def predict_with_factor(prevention_factor, appraisal_factor):
        x_new = base_point.copy()
        for c in prevention_cols:
            if c in x_new.columns:
                x_new[c] = x_new[c] * (1 + prevention_factor)
        for c in appraisal_cols:
            if c in x_new.columns:
                x_new[c] = x_new[c] * (1 + appraisal_factor)
        return float(model.predict(x_new)[0])

    baseline_cost = predict_with_factor(0.0, 0.0)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "prevention_cols": prevention_cols,
        "appraisal_cols": appraisal_cols,
        "baseline_cost": baseline_cost,
        "predict_func": predict_with_factor,
        "metrics": results
    }, None


# -----------------------
# Google Generative AI 챗봇
# -----------------------
def generate_ai_response(user_message, api_key, analysis_summary=""):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    system_prompt = f"""
당신은 중소기업의 Q-COST(예방비용, 평가비용, 내부/외부 실패비용) 분석을 도와주는 AI 컨설턴트입니다.
회귀분석, 랜덤포레스트, XGBoost, SHAP 결과를 바탕으로 품질 비용 구조와 의사결정을 설명합니다.

아래는 현재 데이터 분석 요약입니다:

{analysis_summary}

이 요약을 참고해서 사용자의 질문에 대해:
- 한국어로,
- 최대한 쉽게,
- 숫자와 직관적 표현을 함께 사용해서
설명해 주세요.
"""

    full_prompt = system_prompt + "\n\n사용자 질문:\n" + user_message

    response = model.generate_content(full_prompt)
    return response.text


def get_feature_importance_df(model, feature_names):
    """
    트리 기반 모델(RandomForest, XGBoost 등)의 feature_importances_를
    중요도 순으로 정렬된 DataFrame으로 반환
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def get_shap_importance_df(model, X_train, feature_names):
    """
    SHAP 값을 이용해 각 변수의 평균 |SHAP| (절대값) 중요도를
    숫자 테이블로 반환.
    - shap_values가 list이든, 3차원 이상이든 최대한 2D (n_samples, n_features)로 정리해서 사용.
    """
    sample = X_train

    # 너무 크면 샘플링
    if X_train.shape[0] > 500:
        sample = X_train.sample(500, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # 1) 이진/다중 분류: shap_values가 [class0, class1, ...] 리스트인 경우
    sv = shap_values
    if isinstance(sv, list):
        # 보통 "양성" 클래스(1번)를 많이 보지만,
        # 없으면 첫 번째 클래스 사용
        if len(sv) > 1:
            sv = sv[1]
        else:
            sv = sv[0]

    # 2) numpy array로 통일
    sv = np.array(sv)

    # sv 차원 정리:
    # 보통 (n_samples, n_features)인데, 간혹 (1, n_samples, n_features) 등으로 나올 수 있음
    if sv.ndim > 2:
        # 마지막 축은 feature로 보고, 나머지 축은 전부 sample 차원으로 펼침
        n_features = sv.shape[-1]
        sv = sv.reshape(-1, n_features)

    # 이제 sv는 (n_samples, n_features) 형태라고 가정
    mean_abs_shap = np.mean(np.abs(sv), axis=0)  # -> (n_features,)

    # 혹시라도 남은 차원이 있으면 평탄화
    mean_abs_shap = np.array(mean_abs_shap).ravel()

    # feature_names도 리스트로 캐스팅
    feature_names = list(feature_names)

    # 길이가 안 맞으면 최소 길이에 맞춰 자르기 (방어 코드)
    if len(feature_names) != len(mean_abs_shap):
        n = min(len(feature_names), len(mean_abs_shap))
        feature_names = feature_names[:n]
        mean_abs_shap = mean_abs_shap[:n]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return shap_df



# -----------------------
# Streamlit 앱 시작
# -----------------------
st.set_page_config(page_title="Q-COST AI Chat", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>Q-COST AI Chat</h1>",
    unsafe_allow_html=True
)

# 사이드바: Google API Key & 옵션
with st.sidebar:
    st.header("🔑 설정")
    google_api_key = st.text_input(
        "Google API KEY 입력 (Gemini 사용)",
        type="password",
        help="Google Generative AI(Gemini) API 키를 입력하세요."
    )
    if google_api_key:
        st.success("API 키가 설정되었습니다.", icon="✅")
    else:
        st.info("챗봇 기능을 쓰려면 Google API KEY를 입력하세요.", icon="ℹ️")

    st.markdown("---")
    st.markdown("**파일 업로드 후 분석 칼럼을 지정해주세요**")
    st.markdown("**이후 자동으로**")
    st.markdown("**회귀, 랜덤포레스트, XGBoost, SHAP 중요도 분석을 시작합니다.**")
    st.markdown("**예방/평가비용 시나리오도 적용해볼 수 있습니다.**")

tab_analysis, tab_chat = st.tabs(["자동 Q-COST 분석", "Q-COST AI 대화"])

# 세션 상태에 분석 요약 저장 (챗봇에게 넘기기용)
if "analysis_summary" not in st.session_state:
    st.session_state["analysis_summary"] = ""

with tab_analysis:
    st.subheader("데이터 업로드")

    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드하세요.",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        # 파일 읽기
        if uploaded_file.name.endswith(".csv"):
            df = read_csv_auto(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)


        st.write("#### 원본 데이터 미리보기")
        st.dataframe(df.head())

        # -------------------------
        # Q-COST 컬럼 자동 탐지 + 직접 지정
        # -------------------------
        st.write("### Q-COST 비용 컬럼 지정 (예방/평가)")
        auto_cost_cols = detect_cost_columns(df)
        st.caption("자동으로 찾아본 결과를 기본값으로 넣어두었어요. 필요하면 드롭다운에서 직접 바꿔주세요.")

        all_columns = list(df.columns)

        col1, col2 = st.columns(2)
        with col1:
            prevention_selected = st.multiselect(
                "예방비용 컬럼 선택",
                options=all_columns,
                default=auto_cost_cols["prevention"],
                help="예방 활동, 교육, 설비 개선 등에 쓰이는 비용 컬럼을 선택하세요."
            )
        with col2:
            appraisal_selected = st.multiselect(
                "평가/검사비용 컬럼 선택",
                options=all_columns,
                default=auto_cost_cols["appraisal"],
                help="검사, 시험, 품질점검에 들어가는 비용 컬럼을 선택하세요."
            )

        # 실패비용 타깃 설정: 통합 실패비용 또는 내부/외부 실패비용 합산
        st.write("### 실패비용(타깃) 컬럼 설정")
        st.caption("데이터에 이미 '실패비용'이 한 컬럼으로 있으면 아래에서 바로 선택하고, 내부/외부가 나뉘어 있으면 각각의 컬럼을 선택하세요.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        failure_single_col = st.selectbox(
            "통합 실패비용 컬럼 (있는 경우)",
            options=["(없음)"] + list(numeric_cols),
            index=0
        )

        col3, col4 = st.columns(2)
        with col3:
            failure_internal_cols = st.multiselect(
                "내부 실패비용 컬럼 선택",
                options=all_columns,
                default=auto_cost_cols["internal_failure"],
                help="공정 안에서 발생하는 불량, 재작업, 스크랩 비용 컬럼을 선택하세요."
            )
        with col4:
            failure_external_cols = st.multiselect(
                "외부 실패비용 컬럼 선택",
                options=all_columns,
                default=auto_cost_cols["external_failure"],
                help="고객 클레임, A/S, 리콜 등 외부에서 발생하는 실패비용 컬럼을 선택하세요."
            )

        # 이후 코드에서 사용할 공식 cost_cols (사용자 선택 기준)
        cost_cols = {
            "prevention": prevention_selected,
            "appraisal": appraisal_selected,
            "internal_failure": failure_internal_cols,
            "external_failure": failure_external_cols,
        }

        failure_auto_col = None

        # 내부/외부 실패비용이 각각 선택된 경우 자동 합산 컬럼 생성
        failure_source_cols = [
            c for c in (failure_internal_cols + failure_external_cols)
            if c in numeric_cols
        ]

        # 통합 실패비용 컬럼이 따로 선택되지 않았고, 내부/외부 합산이 가능하면 자동 생성
        if failure_single_col == "(없음)" and failure_source_cols:
            failure_auto_col = "FAILURE_COST_AUTO"
            # 이미 같은 이름이 있으면 덮어쓰지 않고 이름 변경
            suffix = 1
            while failure_auto_col in df.columns:
                failure_auto_col = f"FAILURE_COST_AUTO_{suffix}"
                suffix += 1

            df[failure_auto_col] = df[failure_source_cols].sum(axis=1)

            st.info(
                f"내부/외부 실패비용 컬럼 {failure_source_cols} 를 합산하여 "
                f"'{failure_auto_col}' 칼럼을 자동 생성했습니다. "
                f"타깃 컬럼 선택에서 이 칼럼을 선택하면 '실패비용' 회귀 모델을 학습할 수 있습니다."
            )
        elif failure_single_col != "(없음)":
            # 사용자가 명시적으로 통합 실패비용 컬럼을 지정한 경우
            failure_auto_col = failure_single_col

        # -------------------------
        # 타깃 자동 탐지 및 모델링
        # -------------------------
        st.write("### 타깃(성공/실패 또는 품질 결과) 컬럼 선택")
        # 실패비용 분석 중심이므로, 자동 생성/지정된 실패비용 칼럼이 있으면 이를 타깃 기본값으로 사용
        default_target = None
        if failure_auto_col is not None and failure_auto_col in df.columns:
            default_target = failure_auto_col
        else:
            default_target = detect_target_column(df)

        target_col = st.selectbox(
            "타깃 컬럼을 선택하세요 (실패비용 또는 품질 결과 컬럼)",
            options=["(사용 안 함)"] + list(df.columns),
            index=1 + (list(df.columns).index(default_target) if default_target in df.columns else 0)
        )


        # 숫자 컬럼만 사용 (단순화)
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        if target_col != "(사용 안 함)" and target_col in df.columns:
            # 타깃 처리
            target_series = binarize_target(df[target_col])

            # 타깃이 0/1인지, 연속형인지 확인
            unique_vals = target_series.dropna().unique()
            is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})

            # numeric_df에 타깃 붙이기
            numeric_df[target_col] = target_series

            # 결측치 제거
            data = numeric_df.dropna(subset=[target_col])
            if data.shape[0] < 4:
                st.warning("유효한 데이터 행이 4개 미만입니다. 더 많은 데이터가 있으면 좋습니다.")
            else:
                X = data.drop(columns=[target_col])
                y = data[target_col]
                
                # ---------------------------
                # 🔥 실패비용 관련 컬럼 제거 (중요)
                # ---------------------------
                failure_related_cols = failure_internal_cols + failure_external_cols

                # 자동 생성된 통합 실패비용
                if failure_auto_col is not None:
                    failure_related_cols.append(failure_auto_col)

                # 사용자가 직접 선택한 통합 실패비용
                if failure_single_col != "(없음)":
                    failure_related_cols.append(failure_single_col)

                X = remove_failure_related_features(X, failure_related_cols)
                

                st.write("### 모델 학습 결과")

                if is_binary:
                    st.write("#### (분류) 성공/실패 또는 양품/불량 예측")

                    clf_results, (X_train, X_test, y_train, y_test) = train_models_classification(X, y)

                    # 결과 테이블
                    rows = []
                    for name, r in clf_results.items():
                        rows.append({
                            "Model": name,
                            "Accuracy": r["accuracy"],
                            "F1-score": r["f1"],
                            "ROC-AUC": r["auc"],
                        })
                    st.dataframe(pd.DataFrame(rows).set_index("Model"))

                    # 랜덤포레스트 중요도 + SHAP: 테이블 형태
                    if "RandomForest" in clf_results:
                        rf_model = clf_results["RandomForest"]["model"]
                        st.write("#### 랜덤포레스트 변수 중요도")
                        imp_df = get_feature_importance_df(rf_model, X.columns)
                        if imp_df is not None:
                            st.dataframe(imp_df)

                        st.write("#### 랜덤포레스트 SHAP 중요도")
                        shap_df = get_shap_importance_df(rf_model, X_train, X.columns)
                        st.dataframe(shap_df)

                    # XGBoost도 숫자 테이블로만
                    if XGBOOST_AVAILABLE and "XGBoost" in clf_results:
                        xgb_model = clf_results["XGBoost"]["model"]
                        st.write("#### XGBoost 변수 중요도")
                        imp_df_xgb = get_feature_importance_df(xgb_model, X.columns)
                        if imp_df_xgb is not None:
                            st.dataframe(imp_df_xgb)

                        st.write("#### XGBoost SHAP 중요도")
                        shap_df_xgb = get_shap_importance_df(xgb_model, X_train, X.columns)
                        st.dataframe(shap_df_xgb)

                    # 분석 요약 생성 (챗봇용)
                    summary_lines = ["[분류 모델 요약]"]
                    for name, r in clf_results.items():
                        summary_lines.append(
                            f"- {name}: Accuracy={r['accuracy']:.3f}, F1={r['f1']:.3f}, AUC={r['auc']:.3f}"
                        )
                    summary_lines.append(f"- 사용된 특징 수: {X.shape[1]}")
                    summary_lines.append(f"- 타깃 컬럼: {target_col}")
                    st.session_state["analysis_summary"] = "\n".join(summary_lines)

                else:
                    st.write("#### (회귀) 연속형 품질 지표 예측")

                    reg_results, (X_train, X_test, y_train, y_test) = train_models_regression(X, y)

                    rows = []
                    for name, r in reg_results.items():
                        rows.append({
                            "Model": name,
                            "RMSE": r["rmse"],
                            "R²": r["r2"],
                        })
                    st.dataframe(pd.DataFrame(rows).set_index("Model"))

                    
                    if "RandomForest" in reg_results:
                        rf_model = reg_results["RandomForest"]["model"]
                        st.write("#### 랜덤포레스트 변수 중요도")
                        imp_df = get_feature_importance_df(rf_model, X.columns)
                        if imp_df is not None:
                            st.dataframe(imp_df)

                        st.write("#### 랜덤포레스트 SHAP 중요도")
                        shap_df = get_shap_importance_df(rf_model, X_train, X.columns)
                        st.dataframe(shap_df)

                    if XGBOOST_AVAILABLE and "XGBoost" in reg_results:
                        xgb_model = reg_results["XGBoost"]["model"]
                        st.write("#### XGBoost 변수 중요도")
                        imp_df_xgb = get_feature_importance_df(xgb_model, X.columns)
                        if imp_df_xgb is not None:
                            st.dataframe(imp_df_xgb)

                        st.write("#### XGBoost SHAP 중요도")
                        shap_df_xgb = get_shap_importance_df(xgb_model, X_train, X.columns)
                        st.dataframe(shap_df_xgb)

                    summary_lines = ["[회귀 모델 요약]"]
                    for name, r in reg_results.items():
                        summary_lines.append(
                            f"- {name}: RMSE={r['rmse']:.3f}, R²={r['r2']:.3f}"
                        )
                    summary_lines.append(f"- 사용된 특징 수: {X.shape[1]}")
                    summary_lines.append(f"- 타깃 컬럼: {target_col}")
                    st.session_state["analysis_summary"] = "\n".join(summary_lines)

        else:
            st.info("타깃 컬럼을 '(사용 안 함)'이 아닌 실제 품질 결과 컬럼으로 선택하면 예측 모델이 학습됩니다.")


        # 실패비용 시나리오에서 사용할 타깃 컬럼 결정
        failure_target_col = None
        # 1) 자동 생성/지정된 실패비용 컬럼이 있으면 우선 사용
        if failure_auto_col is not None and failure_auto_col in df.select_dtypes(include=[np.number]).columns:
            failure_target_col = failure_auto_col
        # 2) 그렇지 않고, 현재 타깃 컬럼이 숫자형이면 그 타깃을 실패비용으로 가정
        elif target_col != "(사용 안 함)" and target_col in df.select_dtypes(include=[np.number]).columns:
            failure_target_col = target_col

        st.markdown("---")
        st.write("### 예방/평가비용을 늘렸을 때 실패비용 시나리오")

        scenario_info, err = build_scenario_result(df, cost_cols, failure_col_name=failure_target_col)

        
        if err:
            st.warning(err)
        else:
            base = scenario_info["baseline_cost"]

            colA, colB = st.columns(2)
            with colA:
                prevent_pct = st.slider("예방비용 증가율 (%)", -50, 200, 0, step=1)
            with colB:
                appraisal_pct = st.slider("평가/검사비용 증가율 (%)", -50, 200, 0, step=1)

            new_cost = scenario_info["predict_func"](
                prevent_pct / 100.0,
                appraisal_pct / 100.0
            )

            diff = new_cost - base
            ratio = (new_cost / base - 1) * 100 if base != 0 else 0

            st.write(f"- 기준 예상 실패비용(평균 기준): **{base:,.2f}**")
            st.write(f"- 시나리오 예상 실패비용: **{new_cost:,.2f}**")
            st.write(f"- 변화량: **{diff:,.2f} ({ratio:+.1f}%)**")

            if diff < 0:
                st.success("이 시나리오 일 때, 모델 기준으로 실패비용이 감소하는 경향이 보입니다.")
            elif diff > 0:
                st.warning("이 데이터에서는 예방/평가비용 증가가 오히려 실패비용 증가와 함께 나타날 수도 있습니다. 실제 공정 구조를 다시 점검해 보세요.")
            else:
                st.info("현재 설정에서는 실패비용 변화가 거의 없는 것으로 나타납니다.")

            # 이 부분도 챗봇 분석 요약에 추가
            extra = f"\n\n[시나리오 분석]\n- 기준 실패비용: {base:,.2f}\n- 시나리오 실패비용: {new_cost:,.2f}\n- 변화율: {ratio:+.1f}%"
            st.session_state["analysis_summary"] += extra

    else:
        st.info("CSV 또는 Excel 파일을 업로드하면 자동으로 분석을 시작합니다.")


with tab_chat:
    st.subheader("Q-COST AI 컨설턴트와 대화하기")
    st.caption("**예시) 시나리오 분석 결과 해석해줘.**")
    st.caption("**예시) 모델 성능을 요약해줘.**")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 기존 메시지 표시
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Q-COST, 품질비용, 분석 결과에 대해 무엇이든 물어보세요.")
    if user_input:
        # 사용자 메시지 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if not google_api_key:
                st.warning("Google API KEY를 사이드바에 먼저 입력해 주세요.")
            else:
                # 최신 분석 요약 넘겨서 답변
                analysis_summary = st.session_state.get("analysis_summary", "")
                try:
                    answer = generate_ai_response(
                        user_input,
                        api_key=google_api_key,
                        analysis_summary=analysis_summary
                    )
                except Exception as e:
                    answer = f"Gemini 호출 중 오류가 발생했습니다:\n\n{e}"

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
