import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler

# NLP Kütüphaneleri
import torch
from transformers import AutoTokenizer, AutoModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEP = "=" * 65

# ══════════════════════════════════════════════════════════════════════════════
# 0. AYARLAR VE NLP MODELİ HAZIRLIĞI
# ══════════════════════════════════════════════════════════════════════════════
# Sızıntı (Leakage) testi için anahtar
# True: BERTürk kullanır (Ağır ama potansiyel sızıntı yakalar)
# False: Kural tabanlı NLP kullanır (Hafif ve sızıntıya karşı daha güvenli)
USE_BERT = False

print(SEP)
print(f"0.  NLP MODU: {'BERTürk' if USE_BERT else 'Kural Tabanlı (Rule-based)'}")
print(SEP)

if USE_BERT:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased").to(device)
    bert_model.eval()


    def get_bert_embeddings(text_series, batch_size=32):
        texts = text_series.fillna("").tolist()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(
                    device)
                outputs = bert_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)


    pca_bert = PCA(n_components=15, random_state=42)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  VERİ YÜKLEME VE SABİTLER
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("1.  VERİLER YÜKLENİYOR")
print(SEP)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test_x.csv")

y = train_df["career_success_score"].copy()
train_df = train_df.drop("career_success_score", axis=1)

print(f"  Train : {train_df.shape}   |   Test : {test_df.shape}")

TECH_COLS = [
    "coding_score", "problem_solving_score", "data_structures_score",
    "sql_score", "machine_learning_score", "backend_score",
    "frontend_score", "cloud_score", "devops_score", "project_quality_score",
]
SOFT_COLS = [
    "linkedin_profile_score", "cv_quality_score", "technical_interview_score",
    "hr_interview_score", "communication_score", "teamwork_score",
    "leadership_score", "presentation_score",
]

CAT_COLS = [
    "target_role", "department", "hobby", "preferred_social_media_platform"
]

DROP_COLS_UPDATED = [
    "application_year", "age", "graduation_year", "mentor_feedback_text"
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GELİŞMİŞ ÖZELLİK MÜHENDİSLİĞİ
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame, ref: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    is_train = False
    if ref is None:
        ref = df
        is_train = True

    # ── Ordinal Encoding ─────────────────────────────────────────────────────
    tier_map = {"Tier 1": 4.0, "Tier 2": 3.0, "Tier 3": 2.0, "Tier 4": 1.0}
    if "university_tier" in df.columns:
        df["university_tier"] = df["university_tier"].map(tier_map).fillna(1.0)

    # ── Kategorik Veri Hazırlığı ve Frekans Kodlaması ────────────────────────
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            freq_map = ref[col].value_counts(normalize=True).to_dict()
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)

    # ── Eksik Değer Doldurma ─────────────────────────────────────────────────
    for col in ["english_exam_score", "github_avg_stars", "portfolio_score"]:
        if col in df.columns: df[col] = df[col].fillna(ref[col].mean())
    for col in ["internship_duration_months", "open_source_contribution_count"]:
        if col in df.columns: df[col] = df[col].fillna(0)
    for col in TECH_COLS + SOFT_COLS:
        if col in df.columns: df[col] = df[col].fillna(ref[col].mean())

    # ── Mevcut İstatistikler ve Etkileşimler ─────────────────────────────────────
    df["avg_tech"] = df[TECH_COLS].mean(axis=1)
    df["avg_soft"] = df[SOFT_COLS].mean(axis=1)
    df["std_tech"] = df[TECH_COLS].std(axis=1)
    df["std_soft"] = df[SOFT_COLS].std(axis=1)

    df["tech_soft_diff"] = df["avg_tech"] - df["avg_soft"]
    df["tech_soft_ratio"] = df["avg_tech"] / (df["avg_soft"] + 1e-5)

    df["has_internship"] = (df["internship_duration_months"] > 0).astype(int)
    df["github_log"] = np.log1p(df["github_avg_stars"])
    df["opensource_log"] = np.log1p(df["open_source_contribution_count"])
    df["internship_log"] = np.log1p(df["internship_duration_months"])
    df["experience_combo"] = (df["internship_log"] * 0.5 + df["opensource_log"] * 0.3 + df["github_log"] * 0.2)

    df["uni_x_tech"] = df["university_tier"] * df["avg_tech"]
    df["uni_x_soft"] = df["university_tier"] * df["avg_soft"]

    df["pq_x_ti"] = df["project_quality_score"] * df["technical_interview_score"]
    df["pq_sq"] = df["project_quality_score"] ** 2
    df["min_tech"] = df[TECH_COLS].min(axis=1)
    df["max_tech"] = df[TECH_COLS].max(axis=1)
    df["pq_x_avg_tech"] = df["project_quality_score"] * df["avg_tech"]
    df["total_projects"] = (df["real_client_project_count"]
                            + df["freelance_project_count"]
                            + df["github_repo_count"])

    # ── NLP ÖZELLİK ÇIKARIMI (BERTürk vs Kural Tabanlı) ──────────────────────
    if "mentor_feedback_text" in df.columns:
        df["feedback_length"] = df["mentor_feedback_text"].fillna("").apply(len)
        df["feedback_word_count"] = df["mentor_feedback_text"].fillna("").apply(lambda x: len(x.split()))

        if USE_BERT:
            print(f"  BERTürk çalışıyor... ({len(df)} satır işleniyor)")
            embeddings = get_bert_embeddings(df["mentor_feedback_text"])

            if is_train:
                reduced_embeddings = pca_bert.fit_transform(embeddings)
            else:
                reduced_embeddings = pca_bert.transform(embeddings)

            for i in range(reduced_embeddings.shape[1]):
                df[f"bert_pca_{i}"] = reduced_embeddings[:, i]

        else:
            positive_words = ["güçlü", "dikkat çekici", "umut verici", "yüksek", "potansiyel", "başarılı", "harika",
                              "iyi", "etkili"]
            negative_words = ["geliştirmeli", "çalışması gerek", "daha fazla", "eksik", "yetersiz", "zayıf", "kötü",
                              "dikkat etmeli"]

            df["feedback_pos_count"] = df["mentor_feedback_text"].fillna("").apply(
                lambda x: sum(w in str(x).lower() for w in positive_words)
            )
            df["feedback_neg_count"] = df["mentor_feedback_text"].fillna("").apply(
                lambda x: sum(w in str(x).lower() for w in negative_words)
            )
            df["feedback_sentiment_ratio"] = (df["feedback_pos_count"] + 1) / (df["feedback_neg_count"] + 1)

    # ── Gereksiz Sütunları Kaldır ────────────────────────────────────────────
    df = df.drop(columns=[c for c in DROP_COLS_UPDATED if c in df.columns], errors="ignore")
    if "student_id" in df.columns:
        df = df.set_index("student_id")

    return df


# ── Target Encoding (OOF) ──────────────────────────────────────────────
def target_encode_oof(train_df, test_df, col, target, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_enc = np.zeros(len(train_df))
    global_mean = target.mean()

    for tr_idx, val_idx in kf.split(train_df):
        means = target.iloc[tr_idx].groupby(train_df[col].iloc[tr_idx]).mean()
        oof_enc[val_idx] = train_df[col].iloc[val_idx].map(means).fillna(global_mean)

    test_enc = train_df[col].map(target.groupby(train_df[col]).mean()).fillna(global_mean)
    return oof_enc, test_enc.values


print("\n  Eğitim seti için özellik mühendisliği uygulanıyor...")
X = engineer_features(train_df)
print("  Test seti için özellik mühendisliği uygulanıyor...")
X_tst = engineer_features(test_df, ref=train_df)

for col in ["target_role", "department"]:
    oof_enc, test_enc = target_encode_oof(train_df, test_df, col, y)
    X[f"{col}_target_enc"] = oof_enc
    X_tst[f"{col}_target_enc"] = test_enc

print(f"  Eğitim (işlenmiş): {X.shape}   |   Test (işlenmiş): {X_tst.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  AKILLI ÇAPRAZ DOĞRULAMA YARDIMCISI (OOF PREDICT)
# ══════════════════════════════════════════════════════════════════════════════
N_SPLITS = 5
KF = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


def oof_predict(model, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    oof = np.zeros(len(X))
    t_avg = np.zeros(len(X_test))

    model_name = model.__class__.__name__

    if model_name in ["ExtraTreesRegressor", "Ridge", "XGBRegressor"]:
        X_model = X.select_dtypes(exclude=['object', 'string', 'category'])
        X_test_model = X_test.select_dtypes(exclude=['object', 'string', 'category'])
    else:
        X_model = X.copy()
        X_test_model = X_test.copy()

        if model_name == "LGBMRegressor":
            for col in CAT_COLS:
                if col in X_model.columns:
                    X_model[col] = X_model[col].astype('category')
                    X_test_model[col] = X_test_model[col].astype('category')

    for tr_idx, val_idx in KF.split(X_model):
        X_tr, X_val = X_model.iloc[tr_idx], X_model.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        fit_params = {}

        if model_name == "LGBMRegressor":
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]

        model.fit(X_tr, y_tr, **fit_params)
        oof[val_idx] = model.predict(X_val)
        t_avg += model.predict(X_test_model) / N_SPLITS

    return oof, t_avg, mean_squared_error(y, oof)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  OPTUNA HİPERPARAMETRE OPTİMİZASYONLARI
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("4.  OPTUNA OPTİMİZASYONLARI")
print(SEP)


# XGBoost
def xgb_obj(trial):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    _, _, mse = oof_predict(XGBRegressor(**p), X, y, X_tst)
    return mse


print("\n  [1/4] XGBoost   → 30 deneme ...")
xgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
xgb_study.optimize(xgb_obj, n_trials=30, show_progress_bar=True)


# LightGBM
def lgbm_obj(trial):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 1000, 2000, step=100),
        num_leaves=trial.suggest_int("num_leaves", 20, 150),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state=42, n_jobs=-1, verbose=-1
    )
    _, _, mse = oof_predict(lgb.LGBMRegressor(**p), X, y, X_tst)
    return mse


print("\n  [2/4] LightGBM  → 30 deneme ...")
lgbm_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
lgbm_study.optimize(lgbm_obj, n_trials=30, show_progress_bar=True)


# CatBoost
def cat_obj(trial):
    p = dict(
        iterations=trial.suggest_int("iterations", 300, 1000, step=100),
        depth=trial.suggest_int("depth", 4, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        random_state=42, verbose=0,
        cat_features=CAT_COLS
    )
    _, _, mse = oof_predict(CatBoostRegressor(**p), X, y, X_tst)
    return mse


print("\n  [3/4] CatBoost  → 20 deneme ...")
cat_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
cat_study.optimize(cat_obj, n_trials=20, show_progress_bar=True)


# ExtraTrees
def et_obj(trial):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        random_state=42, n_jobs=-1,
    )
    _, _, mse = oof_predict(ExtraTreesRegressor(**p), X, y, X_tst)
    return mse


print("\n  [4/4] ExtraTrees → 20 deneme ...")
et_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
et_study.optimize(et_obj, n_trials=20, show_progress_bar=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  OOF TAHMİNLERİ ÜRETME VE STACKING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("5.  OOF TAHMİNLERİ ÜRETİLİYOR")
print(SEP)

best_xgb = XGBRegressor(**xgb_study.best_params, random_state=42, n_jobs=-1, tree_method="hist")
best_lgbm = lgb.LGBMRegressor(**lgbm_study.best_params, random_state=42, n_jobs=-1, verbose=-1)
best_cat = CatBoostRegressor(**cat_study.best_params, random_state=42, verbose=0, cat_features=CAT_COLS)
best_et = ExtraTreesRegressor(**et_study.best_params, random_state=42, n_jobs=-1)

MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost", "ExtraTrees"]
BASE_MODELS = [best_xgb, best_lgbm, best_cat, best_et]

oof_cols = []
test_cols = []
base_mse = []

for name, mdl in zip(MODEL_NAMES, BASE_MODELS):
    print(f"\n  {name} Eğitiliyor...", end="", flush=True)
    oof_p, tst_p, mse = oof_predict(mdl, X, y, X_tst)
    oof_cols.append(oof_p)
    test_cols.append(tst_p)
    base_mse.append(mse)
    print(f"  OOF MSE = {mse:.4f}  |  RMSE = {np.sqrt(mse):.4f}")

meta_X_tr = np.column_stack(oof_cols)
meta_X_tst = np.column_stack(test_cols)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  META-LEARNER: Ridge Stacking & Ters-MSE Blend
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("6.  META-LEARNER VE BLEND")
print(SEP)


def meta_obj(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 200.0, log=True)
    pipe = Pipeline([("sc", RobustScaler()), ("r", Ridge(alpha=alpha))])
    scores = cross_val_score(pipe, meta_X_tr, y, cv=KF, scoring="neg_mean_squared_error")
    return -scores.mean()


meta_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
meta_study.optimize(meta_obj, n_trials=40)
meta_mse_oof = meta_study.best_value

meta_pipe = Pipeline([("sc", RobustScaler()), ("r", Ridge(alpha=meta_study.best_params["alpha"]))])
meta_pipe.fit(meta_X_tr, y)
stacked_preds = meta_pipe.predict(meta_X_tst)

inv_w = 1.0 / np.array(base_mse)
weights = inv_w / inv_w.sum()
blend_preds = meta_X_tst @ weights
blend_oof_mse = mean_squared_error(y, meta_X_tr @ weights)

s_inv = 1.0 / meta_mse_oof
b_inv = 1.0 / blend_oof_mse
STACK_W = s_inv / (s_inv + b_inv)
BLEND_W = 1.0 - STACK_W

final_preds = STACK_W * stacked_preds + BLEND_W * blend_preds
final_preds = np.clip(final_preds, 0, 100)

print(f"\n  Blend OOF MSE  = {blend_oof_mse:.4f}")
print(f"  Ridge OOF MSE  = {meta_mse_oof:.4f}")
print(f"  Nihai Ağırlık  → Stacking: {STACK_W:.3f} | Blend: {BLEND_W:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  SONUÇ ÖZETİ VE KAYIT
# ══════════════════════════════════════════════════════════════════════════════
submission = pd.DataFrame({
    "student_id": test_df["student_id"],
    "career_success_score": final_preds,
})
submission.to_csv("submission_2.3.csv", index=False)
print(f"\n  ✓ Tahminler → 'submission_2.3.csv' dosyasına kaydedildi!")
print(SEP)