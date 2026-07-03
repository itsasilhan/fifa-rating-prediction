# FC25 Player Rating Predictor

A segmented machine learning pipeline that predicts a player's overall rating (OVR) in EA Sports FC 25, using a separate model per positional role instead of one generic model for all players.

## Motivation

A single model trained on all players tends to treat every attribute the same way regardless of position, which does not reflect how ratings actually work in football. A center back does not need elite shooting to be rated highly, and a striker does not need elite tackling. At the same time, secondary attributes still matter to some extent, and a player who is exceptional in one profile attribute but critically weak in another should not be overrated.

This project addresses that by:

1. Splitting players into four positional segments.
2. Training a dedicated two-stage model per segment, where profile (primary) attributes explain most of the rating and non-profile (secondary) attributes contribute a smaller, but non-zero, correction.
3. Explicitly modeling the "weak link" effect, where one severely underdeveloped profile attribute pulls the predicted rating down.

## Why a two-stage model instead of feature weighting

Tree-based models such as gradient boosting are invariant to monotonic scaling of a single feature. Multiplying a feature by a constant before training does not change how a tree splits on it, so naive feature weighting has no effect on a gradient boosting model's behavior.

Instead, this project uses a staged (residual) approach:

- **Stage 1 (primary):** a model is trained to predict OVR using only the profile attributes for a given segment (for example, defending, tackling, and height for defenders).
- **Residual:** the difference between the true OVR and the out-of-fold prediction from Stage 1.
- **Stage 2 (secondary):** a second model is trained to predict this residual using the remaining, non-profile attributes.
- **Final prediction:** `primary_prediction + shrink * secondary_prediction`, where `shrink` is a segment-specific factor between 0 and 1 that controls how much influence secondary attributes are allowed to have.

This guarantees that profile attributes dominate the prediction while secondary attributes still contribute a bounded, non-zero effect, which was not achievable through simple feature scaling.

## Segments and profile attributes

| Segment | Positions | Profile (primary) attributes | Secondary shrink factor |
|---|---|---|---|
| Defense | CB, LB, RB, CDM, LWB, RWB | Defending, Interceptions, Defensive Awareness, Standing Tackle, Sliding Tackle, Jumping, Stamina, Aggression, Height | 0.35 |
| Midfield | CM, CAM, LM, RM | Passing, Vision, Short Passing, Long Passing, Curve, Dribbling, Agility, Balance, Reactions, Ball Control, Composure | 0.55 |
| Forward | ST, CF | Pace, Acceleration, Sprint Speed, Shooting, Finishing, Shot Power, Long Shots, Volleys, Penalties, Positioning | 0.35 |
| Winger | LW, RW | Same as Forward, plus Dribbling, Agility, Balance, Reactions, Ball Control, Composure | 0.35 |
| Goalkeeper | GK | GK Diving, GK Handling, GK Kicking, GK Positioning, GK Reflexes, Height | None (single stage only) |

The midfield segment uses a higher shrink factor because pace and physical attributes are still expected to matter close to a normal amount for midfielders, unlike for the other outfield segments.

Wingers are split out from strikers/forwards because, unlike a pure striker, a winger's effectiveness depends heavily on ball control and dribbling ability in tight, wide areas. The dribbling-related attributes (Dribbling, Agility, Balance, Reactions, Ball Control, Composure) are treated as profile attributes for both wingers and central midfielders, but remain secondary for strikers and defenders.

Goalkeepers are modeled independently of outfield attributes entirely, since goalkeeper ratings should only be compared against goalkeeping stats.

## Weak link feature

For each segment, an engineered feature is computed as:

```
weak_link = min(profile attributes) - mean(profile attributes)
```

This becomes more negative the more a single profile attribute lags behind the rest of the profile attributes, which lets the model learn that a severely underdeveloped key attribute should reduce the overall rating, even if other profile attributes are excellent.

## Repository structure

```
train_segmented_fc25.py   Training script: data cleaning, segmentation, two-stage model training, evaluation, and model export
app.py                    Streamlit app for interactive OVR prediction
```

## Data

The training script expects the "EA Sports FC 25 Database, Ratings and Stats" dataset (`all_players.csv`). By default it looks for the file locally, and falls back to downloading it via `kagglehub` if it is not found:

```
nyagami/ea-sports-fc-25-database-ratings-and-stats
```

## Setup

```bash
pip install pandas numpy scikit-learn joblib streamlit kagglehub
```

## Training

```bash
python train_segmented_fc25.py
```

This will:

- Clean and preprocess the dataset (parsing height/weight, encoding categorical fields, handling missing values).
- Assign each player to one of the four segments based on position.
- Train a Stage 1 (primary) and, where applicable, a Stage 2 (secondary) `GradientBoostingRegressor` per segment.
- Print train/test R-squared for each segment, including a comparison between the primary-only model and the final primary-plus-secondary model.
- Save all trained models, scalers, feature lists, and shrink factors as `.pkl` files in the working directory.

### Example output

```
--- Segment: Defense ---
  primary features   : ['DEF', 'Interceptions', 'Def Awareness', 'Standing Tackle', 'Sliding Tackle', 'Jumping', 'Stamina', 'Aggression', 'Height']
  secondary count    : 37  | shrink = 0.35
  Test R2 (primary only)        : 0.9383
  Test R2 (primary + secondary) : 0.9522
```

## Running the app

Once training has produced the `.pkl` files, place them in the same directory as `app.py` (or update `MODELS_DIR` inside the script), then run:

```bash
streamlit run app.py
```

The app lets you:

- Select a player role, which loads the corresponding segment's model.
- Adjust profile attributes with sliders to see their effect on the predicted rating.
- Optionally expand a section to adjust secondary attributes, which default to neutral values if left unchanged.
- View a breakdown of the predicted OVR into its primary and secondary contributions.
- See a warning when a profile attribute is significantly weaker than the others, illustrating the weak link effect.

## Notes and limitations

- Categorical fields such as nation, league, team, and play style are label-encoded using `pandas.factorize`, which assigns integer codes based on the training data only. New, unseen categories at inference time are not meaningfully represented; the app assigns neutral defaults for these fields.
- The residual for Stage 2 training is computed using out-of-fold predictions from Stage 1 (via `cross_val_predict`) to avoid leakage between the two stages.
- Predictions are clipped to the valid OVR range of 1 to 99.
