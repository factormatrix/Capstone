import numpy as np
import pandas as pd
import streamlit as st

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import shap

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# -----------------------
# 유틸 함수들
# -----------------------
def is_tree_model(model):
    """
    RandomForest / XGBoost / LightGBM / CatBoost 같은 트리 기반 모델인지 확인
    """
    tree_types = (RandomForestClassifier, RandomForestRegressor)

    # XGBoost
    try:
        import xgboost as xgb
        tree_types = tree_types + (xgb.XGBClassifier, xgb.XGBRegressor)
    except Exception:
        pass

    # LightGBM
    try:
        import lightgbm as lgb
        tree_types = tree_types + (lgb.LGBMClassifier, lgb.LGBMRegressor)
    except Exception:
        pass

    # CatBoost
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        tree_types = tree_types + (CatBoostClassifier, CatBoostRegressor)
    except Exception:
        pass

    return isinstance(model, tree_types)


def read_csv_auto(file_obj):
    """
    업로드된 CSV 파일을 인코딩을 바꿔 가며 읽는 함수.
    UTF-8이 안 되면 cp949, 그래도 안 되면 ISO-8859-1 시도.
    """
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
        return [c for c in cols if any(k.lower() in str(c).lower() for k in keywords)]

    prevention_cols = match_keywords(["예방", "prevention", "prevention_cost", "prevention cost", "P-COST", "P_COST"])
    appraisal_cols = match_keywords(["평가", "검사", "inspection", "appraisal", "A-COST", "A_COST"])
    internal_failure_cols = match_keywords(["내부", "internal_failure", "internal failure", "IF-COST", "IF_COST"])
    external_failure_cols = match_keywords(["외부", "external_failure", "external failure", "EF-COST", "EF_COST"])

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

    preferred_patterns = [
        "성공", "실패", "합격", "불합격", "불량", "양품",
        "pass_fail", "passfail", "pass", "fail",
        "target", "label", "y"
    ]
    for col in candidates:
        name = str(col).lower()
        if any(p.lower() in name for p in preferred_patterns):
            return col

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
        try:
            return float(v)
        except Exception:
            return np.nan

    s = s.map(_map)

    unique_vals = [u for u in s.dropna().unique()]
    if set(unique_vals).issubset({0, 1}):
        return s

    if len(unique_vals) == 2:
        lo, hi = sorted(unique_vals)
        return s.map(lambda x: 0 if x == lo else (1 if x == hi else np.nan))

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
        max_depth=10,
        min_samples_leaf=3,
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
            max_depth=3,
            learning_rate=0.1,
            min_child_weight=5,
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

        if LIGHTGBM_AVAILABLE:
            try:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=200,
                    num_leaves=20,
                    max_depth=3,
                    min_child_samples=20,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                )
                lgb_model.fit(X_train, y_train)
                y_pred = lgb_model.predict(X_test)
                y_prob = lgb_model.predict_proba(X_test)[:, 1]

                results["LightGBM"] = {
                    "model": lgb_model,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "auc": roc_auc_score(y_test, y_prob),
                }

            except Exception as e:
                msg = str(e)
                if "special JSON characters in feature name" in msg or "feature name" in msg:
                    st.info(
                        "LightGBM에서 피처 이름(문자열) 관련 오류가 발생하여 "
                        "LightGBM 모델은 제외하고 나머지 모델만 사용합니다."
                    )
                else:
                    raise

    if CATBOOST_AVAILABLE:
        cb_model = CatBoostClassifier(
            depth=5,
            learning_rate=0.05,
            iterations=500,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
        )
        cb_model.fit(X_train, y_train)
        y_pred = cb_model.predict(X_test)
        y_prob = cb_model.predict_proba(X_test)[:, 1]

        results["CatBoost"] = {
            "model": cb_model,
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
        max_depth=10,
        min_samples_leaf=3,
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
            max_depth=3,
            learning_rate=0.1,
            min_child_weight=5,
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

    if LIGHTGBM_AVAILABLE:
        try:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=20,
                max_depth=3,
                min_child_samples=20,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            lgb_model.fit(X_train, y_train)
            y_pred = lgb_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            results["LightGBM"] = {
                "model": lgb_model,
                "rmse": rmse,
                "r2": r2_score(y_test, y_pred),
            }

        except Exception as e:
            msg = str(e)
            if "special JSON characters in feature name" in msg or "feature name" in msg:
                st.info(
                    "LightGBM에서 피처 이름(문자열) 관련 오류가 발생하여 "
                    "LightGBM 회귀 모델은 제외하고 나머지 모델만 사용합니다."
                )
            else:
                raise

    if CATBOOST_AVAILABLE:
        cb_model = CatBoostRegressor(
            depth=5,
            learning_rate=0.05,
            iterations=500,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
        )
        cb_model.fit(X_train, y_train)
        y_pred = cb_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        results["CatBoost"] = {
            "model": cb_model,
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


def show_min_q_cost_ratio(df, cost_cols, total_q_cost_col=None):
    """
    데이터 내에서 '전체 Q-COST'가 가장 낮았던 관측치 기준으로
    예방/평가/실패비용 비율을 계산하여 한 문장으로 출력한다.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[0] == 0:
        return

    prevention_cols = [c for c in cost_cols.get("prevention", []) if c in numeric_df.columns]
    appraisal_cols = [c for c in cost_cols.get("appraisal", []) if c in numeric_df.columns]
    internal_failure_cols = [c for c in cost_cols.get("internal_failure", []) if c in numeric_df.columns]
    external_failure_cols = [c for c in cost_cols.get("external_failure", []) if c in numeric_df.columns]

    failure_cols = [c for c in (internal_failure_cols + external_failure_cols) if c in numeric_df.columns]

    if not (prevention_cols or appraisal_cols or failure_cols):
        return

    p = numeric_df[prevention_cols].sum(axis=1) if prevention_cols else pd.Series(0.0, index=numeric_df.index)
    a = numeric_df[appraisal_cols].sum(axis=1) if appraisal_cols else pd.Series(0.0, index=numeric_df.index)
    f = numeric_df[failure_cols].sum(axis=1) if failure_cols else pd.Series(0.0, index=numeric_df.index)

    if total_q_cost_col is not None and total_q_cost_col in numeric_df.columns:
        total_q = numeric_df[total_q_cost_col]
    else:
        total_q = p + a + f

    mask = total_q > 0
    if mask.sum() == 0:
        return

    total_q = total_q[mask]
    p = p[mask]
    a = a[mask]
    f = f[mask]

    idx_min = total_q.idxmin()
    total_min = float(total_q.loc[idx_min])
    if total_min <= 0:
        return

    p_ratio = float(p.loc[idx_min]) / total_min * 100
    a_ratio = float(a.loc[idx_min]) / total_min * 100
    f_ratio = float(f.loc[idx_min]) / total_min * 100

    st.markdown(
        f"💡과거 데이터에서 전체 품질비용이 가장 낮았던 관측 시점의 비용 구조는 "
        f"예방 비용 **{p_ratio:.1f}%**, 평가비용 **{a_ratio:.1f}%**, "
        f"실패비용 **{f_ratio:.1f}%** 입니다."
    )



def build_scenario_result(df, cost_cols, failure_col_name=None, exclude_cols=None):
    """
    예방/평가 비용을 변경했을 때 실패비용 변화 시뮬레이션
    - 실패비용: 사용자가 지정한 통합 실패비용 컬럼 또는 내부/외부 실패비용 합계
    - 타깃: 실패비용 (회귀) 또는 이진 타깃 (분류)
    """
    prevention_cols = cost_cols.get("prevention", [])
    appraisal_cols = cost_cols.get("appraisal", [])
    internal_failure_cols = cost_cols.get("internal_failure", [])
    external_failure_cols = cost_cols.get("external_failure", [])

    df_num = df.select_dtypes(include=[np.number]).copy()

    failure_cols = []
    if failure_col_name is not None and failure_col_name in df_num.columns:
        failure_cols = [failure_col_name]
    else:
        failure_cols = [c for c in (internal_failure_cols + external_failure_cols) if c in df_num.columns]

    if not failure_cols:
        return None, "실패비용(타깃) 수치 컬럼을 찾을 수 없습니다. 실패비용 컬럼을 선택해 주세요."

    df_num["failure_cost"] = df_num[failure_cols].sum(axis=1)

    feature_cols = list(set([c for c in prevention_cols + appraisal_cols if c in df_num.columns]))
    if not feature_cols:
        feature_cols = [c for c in df_num.columns if c != "failure_cost"]

    data = df_num[feature_cols + ["failure_cost"]].dropna()
    if data.shape[0] < 20:
        return None, "시뮬레이션을 하기에는 유효한 데이터가 너무 적습니다."

    X = data[feature_cols]
    y = data["failure_cost"]

    if exclude_cols:
        drop_cols = [c for c in exclude_cols if c in X.columns]
        X = X.drop(columns=drop_cols, errors="ignore")

    failure_related_cols = internal_failure_cols + external_failure_cols
    if failure_col_name is not None:
        failure_related_cols.append(failure_col_name)
    X = remove_failure_related_features(X, failure_related_cols)

    unique_vals = sorted(pd.Series(y).dropna().unique())
    is_binary = unique_vals in ([0, 1], [0], [1])

    if is_binary:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric_value = accuracy_score(y_test, y_pred)
        metric_name = "Accuracy"

        results = {"XGBClassifier": {"model": model, "accuracy": metric_value}}
        best_model_name = "XGBClassifier"

    else:
        results, (X_train, X_test, y_train, y_test) = train_models_regression(X, y)

        if "XGBoost" in results:
            best_model_name = "XGBoost"
        else:
            best_model_name = max(results.keys(), key=lambda m: results[m]["r2"])

        model = results[best_model_name]["model"]
        y_pred = model.predict(X_test)
        metric_value = r2_score(y_test, y_pred)
        metric_name = "R2"

    base_point = X.mean(axis=0).to_frame().T

    baseline_prevention = float(base_point[prevention_cols].sum(axis=1).iloc[0]) if prevention_cols else 0.0
    baseline_appraisal = float(base_point[appraisal_cols].sum(axis=1).iloc[0]) if appraisal_cols else 0.0

    def predict_with_factor(prevention_factor, appraisal_factor):
        x_new = base_point.copy()
        for c in prevention_cols:
            if c in x_new.columns:
                x_new[c] = x_new[c] * (1 + prevention_factor)
        for c in appraisal_cols:
            if c in x_new.columns:
                x_new[c] = x_new[c] * (1 + appraisal_factor)

        if is_binary:
            proba = model.predict_proba(x_new)[0][1]
            return float(proba)
        else:
            return float(model.predict(x_new)[0])

    baseline_cost = predict_with_factor(0.0, 0.0)

    return {
        "model": model,
        "best_model_name": best_model_name,
        "feature_cols": feature_cols,
        "prevention_cols": prevention_cols,
        "appraisal_cols": appraisal_cols,
        "baseline_cost": baseline_cost,
        "baseline_prevention": baseline_prevention,
        "baseline_appraisal": baseline_appraisal,
        "predict_func": predict_with_factor,
        "metrics": results,
        "task_type": "classification" if is_binary else "regression",
        "metric_name": metric_name,
        "metric_value": metric_value,
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


def get_shap_importance_df(model, X_train, feature_names):
    """
    SHAP 값을 이용해 각 변수의 평균 |SHAP| (절대값) 중요도를 숫자 테이블로 반환.
    (UI 수정 금지 파트: 표 렌더링과 연결됨)
    """
    sample = X_train
    if X_train.shape[0] > 500:
        sample = X_train.sample(500, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    sv = shap_values
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]

    sv = np.array(sv)
    if sv.ndim > 2:
        n_features = sv.shape[-1]
        sv = sv.reshape(-1, n_features)

    mean_shap = np.mean(sv, axis=0)
    mean_abs_shap = np.mean(np.abs(sv), axis=0)

    mean_shap = np.array(mean_shap).ravel()
    mean_abs_shap = np.array(mean_abs_shap).ravel()

    feature_names = list(feature_names)
    if len(feature_names) != len(mean_abs_shap):
        n = min(len(feature_names), len(mean_abs_shap), len(mean_shap))
        feature_names = feature_names[:n]
        mean_abs_shap = mean_abs_shap[:n]
        mean_shap = mean_shap[:n]

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_shap": mean_shap,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return shap_df


def render_shap_table(shap_df):
    """
    Streamlit column_config를 활용한 SHAP 표 렌더링 (UI 수정 금지 파트)
    """
    df = shap_df.copy()

    total = df["mean_abs_shap"].sum()
    df["importance_pct"] = df["mean_abs_shap"] / total * 100 if total > 0 else 0.0

    def arrow(v):
        if v > 0:
            return "↑ 증가 방향"
        elif v < 0:
            return "↓ 감소 방향"
        return "→ 영향 거의 없음"

    df["direction_arrow"] = df["mean_shap"].apply(arrow)

    show_cols = ["feature", "importance_pct", "direction_arrow", "mean_shap"]

    st.data_editor(
        df[show_cols],
        column_config={
            "feature": st.column_config.TextColumn("변수명"),
            "importance_pct": st.column_config.ProgressColumn(
                "중요도 (|SHAP| 기준 비율)",
                help="각 변수의 |SHAP| 합 대비 비중(%)",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
            ),
            "direction_arrow": st.column_config.TextColumn(
                "영향 방향",
                help="↑: 타깃(실패/비용) 증가, ↓: 타깃 감소, →: 거의 영향 없음",
            ),
            "mean_shap": st.column_config.NumberColumn(
                "평균 SHAP 값",
                help="양수: 타깃 증가 방향, 음수: 타깃 감소 방향",
                format="%.4f",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )


def compute_feature_r2(X, y):
    """
    각 입력 변수별로 단순 모델을 돌려 R² 점수를 계산.
    (UI 수정 금지 파트: 표 렌더링과 연결됨)
    """
    rows = []
    for col in X.columns:
        if X[col].nunique() < 2:
            continue

        x_col = X[[col]]
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=3,
            n_jobs=-1
        )
        model.fit(x_col, y)
        y_pred = model.predict(x_col)
        r2 = r2_score(y, y_pred)

        rows.append({"feature": col, "R²_single": r2})

    if not rows:
        return pd.DataFrame(columns=["feature", "R²_single"])

    return pd.DataFrame(rows).sort_values("R²_single", ascending=False).reset_index(drop=True)


def render_feature_r2_table(feature_r2_df):
    """
    단일 변수별 R² 점수 표 렌더링 (UI 수정 금지 파트)
    """
    if feature_r2_df.shape[0] == 0:
        st.info("각 변수별 R²를 계산할 수 있는 유효한 숫자형 특징이 부족합니다.")
        return

    df = feature_r2_df.copy()
    df["R2_for_bar"] = df["R²_single"].clip(lower=0.0)

    show_cols = ["feature", "R2_for_bar", "R²_single"]

    st.data_editor(
        df[show_cols],
        column_config={
            "feature": st.column_config.TextColumn("변수명"),
            "R2_for_bar": st.column_config.ProgressColumn(
                "R² (0~1 기준)",
                help="각 변수 하나만 사용한 모델의 R²를 그대로 0~1 구간에서 막대로 표현합니다.\n"
                     "음수 R²는 막대가 0으로 표시되고, 실제 값은 오른쪽 숫자에서 확인하세요.",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "R²_single": st.column_config.NumberColumn(
                "단일 변수 R² (실제 값)",
                help="각 변수 하나만 사용한 모델의 실제 R² 값입니다. (음수 가능)",
                format="%.4f",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )


# -----------------------
# Streamlit 앱 시작
# -----------------------
st.set_page_config(page_title="Q-COST AI", layout="wide")
st.markdown("<h1 style='text-align: center;'>Q-COST AI</h1>", unsafe_allow_html=True)

tab_analysis, tab_chat = st.tabs(["자동 Q-COST 분석", "Q-COST AI 대화"])

if "analysis_summary" not in st.session_state:
    st.session_state["analysis_summary"] = ""

# -----------------------
# 사이드바 UI (업로드 + 설정)
# -----------------------
with st.sidebar:
    st.header("🔑 설정")
    google_api_key = st.sidebar.text_input("Google API Key", type="password")

    if google_api_key and google_api_key.strip():
        st.sidebar.success("API 키가 설정되었습니다.", icon="✅")
    else:
        st.sidebar.info("Google API 키를 입력하세요.")


    st.markdown("---")

    st.header("📤 데이터 업로드")
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일",
        type=["csv", "xlsx", "xls"]
    )

    st.caption("업로드 후, 분석 탭에서 **비용 컬럼 → 타깃 → 제외 컬럼** 순서로 설정하세요.")


# =======================
# 1) 분석 탭
# =======================
with tab_analysis:
    st.subheader("✔ 데이터 확인")

    if uploaded_file is None:
        st.info("사이드바에서 CSV 또는 Excel 파일을 업로드하면 자동으로 분석을 시작합니다.")
        st.stop()

    # 파일 읽기
    if uploaded_file.name.endswith(".csv"):
        df = read_csv_auto(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("#### 원본 데이터 미리보기")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) == 0:
        st.warning("숫자형 컬럼이 없습니다. (회귀/시나리오 분석을 위해 숫자형 비용 컬럼이 필요합니다.)")
        st.stop()

    st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
    # -------------------------
    # Q-COST 컬럼 자동 탐지 + 직접 지정
    # -------------------------
    st.subheader("✔ Q-COST 컬럼 지정 (예방 · 평가 · 실패)")
    auto_cost_cols = detect_cost_columns(df)
    st.caption("자동으로 찾아본 결과를 기본값으로 넣어두었어요. 필요하면 아래에서 직접 바꿔주세요.")

    all_columns = list(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    c1, c2, c3 = st.columns(3)

    with c1:
        prevention_selected = st.multiselect(
            "예방비용(P)",
            options=all_columns,
            default=auto_cost_cols["prevention"],
            help="예방 활동, 교육, 설비 개선 등에 쓰이는 비용 컬럼"
        )

    with c2:
        appraisal_selected = st.multiselect(
            "평가비용(A)",
            options=all_columns,
            default=auto_cost_cols["appraisal"],
            help="검사, 시험, 품질점검에 들어가는 비용 컬럼"
        )

    with c3:
        # 실패비용 기본값(원본 로직 유지): 내부+외부 실패 자동 추천
        failure_default = auto_cost_cols["internal_failure"] + auto_cost_cols["external_failure"]

        # 자동 탐지가 비어있으면 이름 기반 추천(원본 로직 유지)
        if not failure_default:
            for c in numeric_cols:
                name = str(c).lower()
                if any(k in name for k in ["실패", "불량", "failure", "defect"]):
                    failure_default.append(c)

        failure_selected_cols = st.multiselect(
            "실패비용(F)",
            options=list(numeric_cols),
            default=failure_default,
            help="실패비용 컬럼(여러 개 선택 가능)"
        )

    st.caption(
        f"선택 현황 · 예방 {len(prevention_selected)}개 / 평가 {len(appraisal_selected)}개 / 실패 {len(failure_selected_cols)}개"
    )

    cost_cols = {
        "prevention": prevention_selected,
        "appraisal": appraisal_selected,
        "internal_failure": failure_selected_cols,
        "external_failure": [],
    }

    st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
    # -------------------------
    # 타깃 선택 (기존 컬럼 vs 자동 합산)
    # -------------------------
    st.subheader("✔ 타깃(Total Q-COST) 설정")
    target_option = st.radio("종속변수(총 품질비용) 설정",
                         ["기존 컬럼 선택", "자동 계산 (예방+평가+실패)"], horizontal=True)


    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # 합산에 사용할 컬럼들(숫자형만)
    p_cols = [c for c in prevention_selected if c in numeric_df.columns]
    a_cols = [c for c in appraisal_selected if c in numeric_df.columns]
    f_cols = [c for c in failure_selected_cols if c in numeric_df.columns]

    target_col = None
    total_q_cost_col = None

    if target_option == "기존 컬럼 선택":
        target_col = st.selectbox(
            "총 품질비용(Total Q-COST) 컬럼 선택",
            ["(선택)"] + list(numeric_df.columns),
        )
        if target_col == "(선택)":
            target_col = None
    else:
        if p_cols or a_cols or f_cols:
            temp_total_col = "TOTAL_Q_COST"
            suffix = 1
            while temp_total_col in df.columns:
                temp_total_col = f"TOTAL_Q_COST_{suffix}"
                suffix += 1

            numeric_df[temp_total_col] = numeric_df[p_cols + a_cols + f_cols].sum(axis=1)
            df[temp_total_col] = numeric_df[temp_total_col]
            target_col = temp_total_col
            total_q_cost_col = temp_total_col
            st.success(f"'{temp_total_col}' 컬럼이 자동 생성되었습니다.")
        else:
            st.warning("자동 합산을 위해 예방/평가/실패비용 컬럼 중 최소 1개 이상을 선택해 주세요.")
    
    st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
    # -------------------------
    # 메인 모델(분석) 실행
    # -------------------------
    st.subheader("✔ 메인 모델 분석")

    if target_col is None or target_col not in df.columns:
        st.info("타깃 컬럼을 선택하면 메인 모델 학습이 진행됩니다.")
    else:
        main_candidate_cols = [c for c in numeric_df.columns if c != target_col]
        main_exclude_cols = st.multiselect(
            "메인 모델에서 독립변수에서 제외할 컬럼 선택",
            options=main_candidate_cols,
            help="실패율/불량률 등 타깃을 사실상 그대로 반영하는 지표는 제외하는 것을 권장합니다."
        )

        # 타깃 처리 (이진/연속 자동)
        target_series = binarize_target(df[target_col])
        unique_vals = target_series.dropna().unique()
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})

        numeric_df[target_col] = target_series
        data = numeric_df.dropna(subset=[target_col])

        if data.shape[0] < 5:
            st.warning("유효한 데이터 행이 5개 미만입니다. 더 많은 데이터가 있으면 좋습니다.")
        else:
            X = data.drop(columns=[target_col])
            y = data[target_col]

            if main_exclude_cols:
                X = X.drop(columns=[c for c in main_exclude_cols if c in X.columns], errors="ignore")

            # (기존 로직 유지) 필요시 실패비용 관련 컬럼 제거하여 leakage 방지
            failure_related_cols = list(failure_selected_cols)
            remove_failure_features = True
            if total_q_cost_col is not None and target_col == total_q_cost_col:
                remove_failure_features = False
            if remove_failure_features and failure_related_cols:
                X = remove_failure_related_features(X, failure_related_cols)

            st.write("### 모델 학습 결과")

            if is_binary:
                st.write("#### (분류) 성공/실패 또는 양품/불량 예측")

                clf_results, (X_train, X_test, y_train, y_test) = train_models_classification(X, y)

                best_overall_name = max(clf_results.keys(), key=lambda m: clf_results[m]["auc"])
                best_overall_model = clf_results[best_overall_name]["model"]

                tree_candidates = [name for name, res in clf_results.items() if is_tree_model(res["model"])]

                shap_model_name = None
                if is_tree_model(best_overall_model):
                    shap_model_name = best_overall_name
                elif tree_candidates:
                    shap_model_name = max(tree_candidates, key=lambda m: clf_results[m]["auc"])

                if shap_model_name is not None:
                    shap_model = clf_results[shap_model_name]["model"]
                    if shap_model_name == best_overall_name:
                        title = f"#### SHAP 변수 중요도 (최고 성능 모델: {shap_model_name})"
                    else:
                        title = (
                            "#### SHAP 변수 중요도 "
                            f"(Tree 모델 기준: {shap_model_name}, "
                            f"전체 최고 모델(AUC): {best_overall_name})"
                        )
                    st.write(title)

                    shap_df_best = get_shap_importance_df(shap_model, X_train, X.columns)
                    render_shap_table(shap_df_best)
                else:
                    st.info("트리 기반 모델이 없어 SHAP 변수 중요도를 계산하지 않습니다. 대신 성능 지표만 참고해주세요.")

                summary_lines = ["[분류 모델 요약]"]
                for name, r in clf_results.items():
                    summary_lines.append(
                        f"- {name}: Accuracy={r['accuracy']:.3f}, F1={r['f1']:.3f}, AUC={r['auc']:.3f}"
                    )
                summary_lines.append(f"- 전체 최고 성능 모델(AUC 기준): {best_overall_name}")
                if shap_model_name is not None and shap_model_name != best_overall_name:
                    summary_lines.append(f"- SHAP 변수 중요도는 트리 기반 모델 '{shap_model_name}' 기준으로 산출")
                summary_lines.append(f"- 사용된 특징 수: {X.shape[1]}")
                summary_lines.append(f"- 타깃 컬럼: {target_col}")
                st.session_state["analysis_summary"] = "\n".join(summary_lines)

            else:
                st.write("#### (회귀) Total Q-COST 예측")

                reg_results, (X_train, X_test, y_train, y_test) = train_models_regression(X, y)

                # ✅ UI 수정 금지
                st.write("#### 단일 변수별 R² 점수")
                feature_r2_df = compute_feature_r2(X, y)
                render_feature_r2_table(feature_r2_df)

                best_overall_name = max(reg_results.keys(), key=lambda m: reg_results[m]["r2"])
                best_overall_model = reg_results[best_overall_name]["model"]

                tree_candidates = [name for name, res in reg_results.items() if is_tree_model(res["model"])]

                shap_model_name = None
                if is_tree_model(best_overall_model):
                    shap_model_name = best_overall_name
                elif tree_candidates:
                    shap_model_name = max(tree_candidates, key=lambda m: reg_results[m]["r2"])

                if shap_model_name is not None:
                    shap_model = reg_results[shap_model_name]["model"]
                    if shap_model_name == best_overall_name:
                        title = f"#### SHAP 변수 중요도 (최고 성능 모델: {shap_model_name})"
                    else:
                        title = (
                            "#### SHAP 변수 중요도 "
                            f"(Tree 모델 기준: {shap_model_name}, "
                            f"전체 최고 모델(R²): {best_overall_name})"
                        )
                    st.write(title)

                    shap_df_best = get_shap_importance_df(shap_model, X_train, X.columns)
                    render_shap_table(shap_df_best)
                else:
                    st.info("트리 기반 회귀 모델이 없어 SHAP 변수 중요도를 계산하지 않습니다. 대신 R², RMSE 지표를 참고해주세요.")

                summary_lines = ["[회귀 모델 요약]"]
                for name, r in reg_results.items():
                    summary_lines.append(f"- {name}: RMSE={r['rmse']:.3f}, R²={r['r2']:.3f}")
                summary_lines.append(f"- 전체 최고 성능 모델(R² 기준): {best_overall_name}")
                if shap_model_name is not None and shap_model_name != best_overall_name:
                    summary_lines.append(f"- SHAP 변수 중요도는 트리 기반 모델 '{shap_model_name}' 기준으로 산출")
                summary_lines.append(f"- 사용된 특징 수: {X.shape[1]}")
                summary_lines.append(f"- 타깃 컬럼: {target_col}")
                st.session_state["analysis_summary"] = "\n".join(summary_lines)

    # -------------------------
    # 참고: 과거 최소 비용구조
    # -------------------------
    show_min_q_cost_ratio(df, cost_cols, total_q_cost_col=total_q_cost_col)

    st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)

    # -------------------------
    # 시나리오 (로직 절대 변경 금지: 출력 UI만 변경)
    # -------------------------
    st.markdown("---")
    st.subheader("🔎 예방/평가비용 변화 시나리오 (실패비용/총비용 변화 확인)")

    # 시나리오 회귀에서 독립변수에서 제외할 컬럼 선택
    scenario_candidate_cols = list(df.select_dtypes(include=[np.number]).columns)

    scenario_exclude_cols = st.multiselect(
        "시나리오 모델에서 독립변수에서 제외할 컬럼 선택 (실패율/불량률 등)",
        options=scenario_candidate_cols,
        help="실패비용과 너무 직접적으로 연결된 지표를 제외하면 시나리오 모델 해석력이 높아집니다."
    )

    # (기존 로직 그대로) failure_target_col 결정: 선택된 실패비용 컬럼들을 활용
    failure_target_col = None
    # 1) 실패비용 컬럼이 1개면 그 컬럼을 사용
    if len(failure_selected_cols) == 1 and failure_selected_cols[0] in numeric_cols:
        failure_target_col = failure_selected_cols[0]
    # 2) 여러 개면 None으로 두고 build_scenario_result 내부 합산 사용
    else:
        failure_target_col = None

    scenario_info, err = build_scenario_result(
        df,
        cost_cols,
        failure_col_name=failure_target_col,
        exclude_cols=scenario_exclude_cols
    )

    if err:
        st.warning(err)
    else:
        base = float(scenario_info["baseline_cost"])
        task_type = scenario_info.get("task_type", "regression")

        def get_min_observed_shares(_df, prevention_cols, appraisal_cols, _failure_target_col):
            df_num = _df.select_dtypes(include=[np.number]).copy()
            p_cols2 = [c for c in prevention_cols if c in df_num.columns]
            a_cols2 = [c for c in appraisal_cols if c in df_num.columns]

            # 실패비용은: 단일 컬럼이면 그 컬럼, 아니면 선택된 실패비용 컬럼 합산
            if _failure_target_col and _failure_target_col in df_num.columns:
                f_cols2 = [_failure_target_col]
            else:
                f_cols2 = [c for c in failure_selected_cols if c in df_num.columns]

            p = df_num[p_cols2].sum(axis=1) if p_cols2 else pd.Series(0.0, index=df_num.index)
            a = df_num[a_cols2].sum(axis=1) if a_cols2 else pd.Series(0.0, index=df_num.index)
            f = df_num[f_cols2].sum(axis=1) if f_cols2 else pd.Series(0.0, index=df_num.index)

            tot = p + a + f
            mask = tot > 0
            if mask.sum() == 0:
                return None

            idx = tot[mask].idxmin()
            tot_min = float(tot.loc[idx])
            return {
                "p_share": float(p.loc[idx]) / tot_min * 100.0,
                "a_share": float(a.loc[idx]) / tot_min * 100.0,
                "f_share": float(f.loc[idx]) / tot_min * 100.0,
            }

        if task_type == "classification":
            colA, colB = st.columns(2)
            with colA:
                prevent_pct = st.slider("예방비용 증가율 (%)", -50, 200, 0, step=1)
            with colB:
                appraisal_pct = st.slider("평가/검사비용 증가율 (%)", -50, 200, 0, step=1)

            new_cost = scenario_info["predict_func"](prevent_pct / 100.0, appraisal_pct / 100.0)

            base_prob = base
            new_prob = float(new_cost)
            prob_diff = (new_prob - base_prob) * 100  # %p

            c1, c2, c3 = st.columns(3)
            c1.metric("기준 실패 발생 확률", f"{base_prob:.3f}")
            c2.metric("현재 실패 발생 확률", f"{new_prob:.3f}")
            c3.metric("변화", f"{prob_diff:+.2f}%p")

        else:
            # ====== (기존 회귀 시나리오 로직 그대로) ======
            base_prev = float(scenario_info.get("baseline_prevention", 0.0))
            base_appr = float(scenario_info.get("baseline_appraisal", 0.0))
            base_fail = float(base)

            base_total_q = base_prev + base_appr + base_fail
            if base_total_q <= 0:
                st.warning("기준 총 품질비용(예방+평가+실패)이 0 이하라 시나리오 계산이 어렵습니다.")
            else:
                obs = get_min_observed_shares(
                    df,
                    scenario_info.get("prevention_cols", []),
                    scenario_info.get("appraisal_cols", []),
                    failure_target_col
                )
                if obs is None:
                    obs_p = (base_prev / base_total_q * 100.0) if base_total_q > 0 else 10.0
                    obs_a = (base_appr / base_total_q * 100.0) if base_total_q > 0 else 10.0
                else:
                    obs_p, obs_a = obs["p_share"], obs["a_share"]

                def clamp(x, lo, hi):
                    return max(lo, min(hi, x))

                P_FACTOR_MIN, P_FACTOR_MAX = -0.50, 2.00
                A_FACTOR_MIN, A_FACTOR_MAX = -0.50, 2.00

                def predict_from_shares(p_share, a_share):
                    if p_share + a_share >= 100.0:
                        return None

                    p_new_raw = base_total_q * (p_share / 100.0)
                    a_new_raw = base_total_q * (a_share / 100.0)

                    p_factor = (p_new_raw / base_prev - 1.0) if base_prev > 0 else 0.0
                    a_factor = (a_new_raw / base_appr - 1.0) if base_appr > 0 else 0.0
                    p_factor = clamp(p_factor, P_FACTOR_MIN, P_FACTOR_MAX)
                    a_factor = clamp(a_factor, A_FACTOR_MIN, A_FACTOR_MAX)

                    p_new = base_prev * (1.0 + p_factor)
                    a_new = base_appr * (1.0 + a_factor)

                    fail_pred = float(scenario_info["predict_func"](p_factor, a_factor))
                    total_pred = float(p_new + a_new + fail_pred)

                    if total_pred <= 0:
                        return None

                    fail_share_pred = fail_pred / total_pred * 100.0
                    return fail_pred, total_pred, fail_share_pred

                BAND_P = 20.0
                BAND_A = 20.0
                step = 1.0

                p_min = max(0.0, obs_p - BAND_P)
                p_max = min(95.0, obs_p + BAND_P)
                a_min_base = max(0.0, obs_a - BAND_A)
                a_max_base = min(95.0, obs_a + BAND_A)

                LAMBDA = 0.02

                best_score = float("inf")
                best_p_share = None
                best_a_share = None

                p = p_min
                while p <= p_max + 1e-9:
                    a_max = min(a_max_base, 99.0 - p)
                    a = a_min_base
                    while a <= a_max + 1e-9:
                        out = predict_from_shares(p, a)
                        if out is not None:
                            _, total_pred, _ = out
                            reg = LAMBDA * ((p - obs_p) ** 2 + (a - obs_a) ** 2)
                            score = total_pred + reg
                            if score < best_score:
                                best_score = score
                                best_p_share = float(p)
                                best_a_share = float(a)
                        a += step
                    p += step

                if best_p_share is None:
                    st.warning("최적 비율 탐색에 실패했습니다. 데이터/컬럼 선택을 확인해 주세요.")
                else:
                    st.markdown(
                        f"💡 **(예측 기반) 총 품질비용이 최소가 되는 비율(총=100%)**: "
                        f"예방 **{best_p_share:.0f}%**, 평가 **{best_a_share:.0f}%**"
                    )

                    
                    if "scenario_p_share" not in st.session_state:
                        st.session_state["scenario_p_share"] = float(best_p_share)
                    if "scenario_a_share" not in st.session_state:
                        st.session_state["scenario_a_share"] = float(best_a_share)

                    colA, colB = st.columns(2)

                    with colA:
                        p_share = st.slider(
                            "예방비용 비율",
                            0.0, 95.0,
                            step=1.0,
                            key="scenario_p_share"
                        )

                    remaining = max(1.0, 99.0 - float(p_share))
                    if float(st.session_state.get("scenario_a_share", 0.0)) > remaining:
                        st.session_state["scenario_a_share"] = remaining

                    with colB:
                        a_share = st.slider(
                            "평가비용 비율",
                            0.0, remaining,
                            step=1.0,
                            key="scenario_a_share"  
                        )
                        

                    out = predict_from_shares(p_share, a_share)
                    f_share_input = 100.0 - float(p_share) - float(a_share)  # ✅ 총=100 구성 잔여 실패비율
                    st.caption(f"총=100 구성 · 예방 {p_share:.0f}% / 평가 {a_share:.0f}% / 실패 {f_share_input:.0f}%")

                    if out is None:
                        st.warning("현재 비율 조합은 계산할 수 없습니다. (p+a가 100%에 너무 가까움/비현실 구간)")
                    else:
                        new_fail, new_total, new_fail_share = out

                        fail_change_pct = (new_fail / base_fail - 1.0) * 100.0 if base_fail != 0 else 0.0
                        total_change_pct = (new_total / base_total_q - 1.0) * 100.0 if base_total_q != 0 else 0.0

                        base_fail_share = (base_fail / base_total_q) * 100.0 if base_total_q != 0 else 0.0
                        fail_share_delta = float(new_fail_share) - float(base_fail_share)  # 퍼센트포인트 변화(pp)

                        #st.metric로 직관 표시
                        def arrow(v: float) -> str:
                            if v > 0:
                                return "📈"
                            elif v < 0:
                                return "📉"
                            return "⏺️"

                        m1, m2, m3= st.columns(3)

                        m1.metric(
                            label=f"{arrow(fail_change_pct)} 실패비용 변화",
                            value=f"{fail_change_pct:+.1f}%",
                            delta=float(fail_change_pct),
                            delta_color="inverse"
                        )

                        m2.metric(
                            label=f"{arrow(total_change_pct)} 전체 품질비용 변화",
                            value=f"{total_change_pct:+.1f}%",
                            delta=float(total_change_pct),
                            delta_color="inverse"
                        )

                        # ✅ 총=100 구성 기준 실패비율(잔여) — 항상 합계 100이 되게 보이는 값
                        m3.metric(
                            label="🧩 잔여 실패비용 비율",
                            value=f"{f_share_input:.1f}%"
                        )

                        st.markdown(
                            """
                            <small style="color: #AAAAAA;">
                            ※ <b>안내</b><br>
                            ‘실패비용 변화(%)’와 ‘전체 품질비용 변화(%)’는 과거 데이터 평균을 기준으로 한 변화율입니다.<br>
                            ‘잔여 실패비용 비율’은 입력한 비용 구성 비율입니다.<br>
                            본 ‘최적 비율’은 비현실 구간(p+a가 100%에 매우 근접) 및
                            데이터 외삽 위험 구간을 제외한 범위 내에서 탐색된 최소값입니다.
                            </small>
                            """,
                            unsafe_allow_html=True
                        )




# =======================
# 2) 챗봇 탭
# =======================
with tab_chat:
    st.subheader("Q-COST AI와 대화하기")
    st.caption("예: 시나리오 분석 결과 해석해줘 / 모델 성능을 요약해줘")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Q-COST, 품질비용, 분석 결과에 대해 무엇이든 물어보세요.")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if not google_api_key:
                st.warning("Google API KEY를 사이드바에 먼저 입력해 주세요.")
            else:
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
