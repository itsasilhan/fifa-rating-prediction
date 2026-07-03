"""
Streamlit app for OVR prediction using the segmented model.

How to run:
    1) First run train_segmented_fc25.py - it creates the files
       model_primary_*.pkl, scaler_primary_*.pkl, features_primary_*.pkl,
       model_secondary_*.pkl (except GK), scaler_secondary_*.pkl,
       features_secondary_*.pkl, shrink_*.pkl
    2) Put all these .pkl files in the same folder as this script
       (or change MODELS_DIR below)
    3) streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = "." 

st.set_page_config(page_title="FC25 OVR Predictor", layout="centered")
SEGMENT_CONFIG = {
    "Defense (CB / LB / RB / CDM)": {
        "key": "Defense",
        "positions": ["CB", "LB", "RB", "CDM", "LWB", "RWB"],
        "primary": ["DEF", "Interceptions", "Def Awareness", "Standing Tackle",
                    "Sliding Tackle", "Jumping", "Stamina", "Aggression", "Height"],
        "has_secondary": True,
    },
    "Midfield (CM / CAM / LM / RM)": {
        "key": "Midfield",
        "positions": ["CM", "CAM", "LM", "RM"],
        "primary": ["PAS", "Vision", "Short Passing", "Long Passing", "Curve",
                     "Dribbling", "Agility", "Balance", "Reactions", "Ball Control", "Composure"],
        "has_secondary": True,
    },
    "Forward (ST / CF)": {
        "key": "Forward",
        "positions": ["ST", "CF"],
        "primary": ["PAC", "Acceleration", "Sprint Speed", "SHO", "Finishing",
                     "Shot Power", "Long Shots", "Volleys", "Penalties", "Positioning"],
        "has_secondary": True,
    },
    "Winger (LW / RW)": {
        "key": "Winger",
        "positions": ["LW", "RW"],
        "primary": ["PAC", "Acceleration", "Sprint Speed", "SHO", "Finishing",
                     "Shot Power", "Long Shots", "Volleys", "Penalties", "Positioning",
                     "Dribbling", "Agility", "Balance", "Reactions", "Ball Control", "Composure"],
        "has_secondary": True,
    },
    "Goalkeeper (GK)": {
        "key": "Goalkeeper",
        "positions": ["GK"],
        "primary": ["GK Diving", "GK Handling", "GK Kicking", "GK Positioning",
                     "GK Reflexes", "Height"],
        "has_secondary": False,
    },
}

FIELD_ATTRS = [
    "PAC", "SHO", "PAS", "DRI", "DEF", "PHY",
    "Acceleration", "Sprint Speed", "Positioning", "Finishing", "Shot Power",
    "Long Shots", "Volleys", "Penalties", "Vision", "Crossing",
    "Free Kick Accuracy", "Short Passing", "Long Passing", "Curve", "Dribbling",
    "Agility", "Balance", "Reactions", "Ball Control", "Composure",
    "Interceptions", "Heading Accuracy", "Def Awareness", "Standing Tackle",
    "Sliding Tackle", "Jumping", "Stamina", "Strength", "Aggression",
]

LABELS = {
    "PAC": "Pace (PAC)", "SHO": "Shooting (SHO)", "PAS": "Passing (PAS)",
    "DRI": "Dribbling (DRI)", "DEF": "Defending (DEF)", "PHY": "Physical (PHY)",
    "Acceleration": "Acceleration", "Sprint Speed": "Sprint Speed",
    "Positioning": "Positioning", "Finishing": "Finishing",
    "Shot Power": "Shot Power", "Long Shots": "Long Shots",
    "Volleys": "Volleys", "Penalties": "Penalties", "Vision": "Vision",
    "Crossing": "Crossing", "Free Kick Accuracy": "Free Kick Accuracy",
    "Short Passing": "Short Passing", "Long Passing": "Long Passing",
    "Curve": "Curve", "Dribbling": "Dribbling", "Agility": "Agility",
    "Balance": "Balance", "Reactions": "Reactions", "Ball Control": "Ball Control",
    "Composure": "Composure", "Interceptions": "Interceptions",
    "Heading Accuracy": "Heading Accuracy", "Def Awareness": "Defensive Awareness",
    "Standing Tackle": "Standing Tackle", "Sliding Tackle": "Sliding Tackle",
    "Jumping": "Jumping", "Stamina": "Stamina", "Strength": "Strength",
    "Aggression": "Aggression", "Height": "Height (cm)", "Weight": "Weight (kg)",
    "Age": "Age", "Weak foot": "Weak Foot (1-5)", "Skill moves": "Skill Moves (1-5)",
    "GK Diving": "GK Diving", "GK Handling": "GK Handling",
    "GK Kicking": "GK Kicking", "GK Positioning": "GK Positioning",
    "GK Reflexes": "GK Reflexes",
}


@st.cache_resource
def load_segment_artifacts(seg_key: str, has_secondary: bool):
    artifacts = {
        "model_primary": joblib.load(os.path.join(MODELS_DIR, f"model_primary_{seg_key}.pkl")),
        "scaler_primary": joblib.load(os.path.join(MODELS_DIR, f"scaler_primary_{seg_key}.pkl")),
        "features_primary": joblib.load(os.path.join(MODELS_DIR, f"features_primary_{seg_key}.pkl")),
        "shrink": joblib.load(os.path.join(MODELS_DIR, f"shrink_{seg_key}.pkl")),
    }
    if has_secondary:
        artifacts["model_secondary"] = joblib.load(os.path.join(MODELS_DIR, f"model_secondary_{seg_key}.pkl"))
        artifacts["scaler_secondary"] = joblib.load(os.path.join(MODELS_DIR, f"scaler_secondary_{seg_key}.pkl"))
        artifacts["features_secondary"] = joblib.load(os.path.join(MODELS_DIR, f"features_secondary_{seg_key}.pkl"))
    return artifacts


def build_secondary_row(secondary_feats: list[str], user_vals: dict) -> pd.DataFrame:
    """Any secondary feature not present in the form gets a neutral default."""
    row = {}
    for f in secondary_feats:
        if f in user_vals:
            row[f] = user_vals[f]
        elif f in FIELD_ATTRS:
            row[f] = 60.0  # neutral default for optional stats
        else:
            row[f] = 0.0  # categorical (Nation/League/Team/...) and Preferred foot dummy -> neutral
    return pd.DataFrame([row])[secondary_feats]


def predict_ovr(seg_key: str, has_secondary: bool, primary_vals: dict, secondary_vals: dict) -> tuple[float, float, float]:
    art = load_segment_artifacts(seg_key, has_secondary)
    primary_feats = art["features_primary"]  # includes weak_link_primary as the last entry

    base_primary_cols = [f for f in primary_feats if f != "weak_link_primary"]
    prim_row = pd.DataFrame([primary_vals])[base_primary_cols]
    weak_link = prim_row.to_numpy(dtype=float).min() - prim_row.to_numpy(dtype=float).mean()
    prim_row["weak_link_primary"] = weak_link
    prim_row = prim_row[primary_feats]

    pred_primary = art["model_primary"].predict(art["scaler_primary"].transform(prim_row))[0]

    pred_secondary_contrib = 0.0
    if has_secondary and art["shrink"]:
        sec_row = build_secondary_row(art["features_secondary"], secondary_vals)
        pred_secondary = art["model_secondary"].predict(art["scaler_secondary"].transform(sec_row))[0]
        pred_secondary_contrib = art["shrink"] * pred_secondary

    final_pred = float(np.clip(pred_primary + pred_secondary_contrib, 1, 99))
    return final_pred, float(pred_primary), float(pred_secondary_contrib)


st.title("FC25 - OVR Prediction by Segment")
st.caption("Segmented model: profile attributes drive most of the rating, "
           "non-profile attributes contribute less, but never nothing.")

segment_label = st.selectbox("Select player role", list(SEGMENT_CONFIG.keys()))
cfg = SEGMENT_CONFIG[segment_label]
seg_key = cfg["key"]

models_missing = not os.path.exists(os.path.join(MODELS_DIR, f"model_primary_{seg_key}.pkl"))
if models_missing:
    st.error(
        f"Could not find model_primary_{seg_key}.pkl in folder `{MODELS_DIR}`. "
        "Run train_segmented_fc25.py first and place the .pkl files next to this app."
    )
    st.stop()

st.subheader("Profile attributes")
st.caption("These attributes drive most of the rating for the selected role.")

primary_vals = {}
cols = st.columns(2)
for i, feat in enumerate(cfg["primary"]):
    label = LABELS.get(feat, feat)
    default = 175.0 if feat == "Height" else 70
    min_v, max_v = (150.0, 210.0) if feat == "Height" else (1, 99)
    with cols[i % 2]:
        primary_vals[feat] = st.slider(label, min_value=min_v, max_value=max_v,
                                        value=default, key=f"prim_{feat}")

secondary_vals = {}
if cfg["has_secondary"]:
    with st.expander("Non-profile attributes (optional, weaker influence)"):
        st.caption("If left untouched, neutral default values (60) are used.")
        other_common = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]
        other_common = [f for f in other_common if f not in cfg["primary"]]
        cols2 = st.columns(2)
        for i, feat in enumerate(other_common):
            with cols2[i % 2]:
                secondary_vals[feat] = st.slider(LABELS.get(feat, feat), 1, 99, 65,
                                                  key=f"sec_{feat}")
        age = st.slider(LABELS["Age"], 16, 45, 24, key="sec_age")
        weight = st.slider(LABELS["Weight"], 55.0, 110.0, 75.0, key="sec_weight")
        weak_foot = st.slider(LABELS["Weak foot"], 1, 5, 3, key="sec_weakfoot")
        skill_moves = st.slider(LABELS["Skill moves"], 1, 5, 3, key="sec_skillmoves")
        secondary_vals.update({
            "Age": age, "Weight": weight, "Weak foot": weak_foot, "Skill moves": skill_moves,
        })

st.divider()

if st.button("Calculate OVR", type="primary", use_container_width=True):
    final_pred, pred_primary, pred_secondary_contrib = predict_ovr(
        seg_key, cfg["has_secondary"], primary_vals, secondary_vals
    )

    st.metric("Predicted OVR", f"{final_pred:.1f}")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Contribution from profile attributes", f"{pred_primary:.1f}")
    with c2:
        st.metric("Contribution from non-profile attributes", f"{pred_secondary_contrib:+.1f}")

    prim_values = list(primary_vals.values())
    weakest_idx = int(np.argmin(prim_values))
    weakest_feat = cfg["primary"][weakest_idx]
    if prim_values[weakest_idx] < (sum(prim_values) / len(prim_values)) - 10:
        st.warning(
            f"'{LABELS.get(weakest_feat, weakest_feat)}' is notably lower than the "
            "other profile attributes - this lowers the final rating (the 'weak link' effect)."
        )

st.divider()
st.caption(
    "The model is trained separately for each role: defenders are rated mainly on "
    "defensive attributes and height, midfielders on passing and vision, "
    "attackers on pace and shooting, and goalkeepers exclusively on goalkeeping stats."
)