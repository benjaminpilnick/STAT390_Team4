import optuna
import os
import json

# ── UPDATE THESE TWO PATHS ──────────────────────────────────────────────────
DB_PATH    = "sqlite:///home/bdp1083/Presentation_4_code_Ben/Updated_OPTUNA/optuna_5fold_runs/optuna_5fold.db"   # path to your .db file
OUTPUT_DIR = "/home/bdp1083/Presentation_4_code_Ben/Updated_OPTUNA/optuna_5fold_runs"                           # where outputs will be saved
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load study
study = optuna.create_study(
    study_name="mil_5fold_sobol16_tpe64",
    storage=DB_PATH,
    direction="minimize",
    load_if_exists=True,
)

# Print summary
print(f"Total trials:    {len(study.trials)}")
print(f"Completed:       {sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)}")
print(f"Pruned:          {sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}")
print(f"Best trial #:    {study.best_trial.number}")
print(f"Best value:      {study.best_value:.6f}")
print(f"Best params:")
for k, v in study.best_params.items():
    if isinstance(v, float) and v < 0.01:
        print(f"  {k}: {v:.3e}")
    else:
        print(f"  {k}: {v:.6f}")

# Save CSV
df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
csv_path = os.path.join(OUTPUT_DIR, "optuna_trials.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved trials CSV: {csv_path}")

# Save summary JSON
summary = {
    "study_name": study.study_name,
    "n_trials_total": len(study.trials),
    "n_trials_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
    "n_trials_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    "best_trial_number": study.best_trial.number,
    "best_value": study.best_value,
    "best_params": study.best_params,
    "best_user_attrs": study.best_trial.user_attrs,
}
json_path = os.path.join(OUTPUT_DIR, "study_summary.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary JSON: {json_path}")

# Save plots
plots = {
    "optimization_history":  optuna.visualization.plot_optimization_history,
    "param_importances":     optuna.visualization.plot_param_importances,
    "parallel_coordinate":   optuna.visualization.plot_parallel_coordinate,
    "contour":               optuna.visualization.plot_contour,
    "slice":                 optuna.visualization.plot_slice,
}

for name, plot_fn in plots.items():
    try:
        fig = plot_fn(study)
        path = os.path.join(OUTPUT_DIR, f"{name}.html")
        fig.write_html(path)
        print(f"Saved plot: {path}")
    except Exception as e:
        print(f"Could not generate {name} plot: {e}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")