"""Streamlit dashboard: AI-Based Performance Analysis and Execution Time Prediction of Serial and
Parallel Sorting Algorithms on Multicore Processors.

Launch:  .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import streamlit as st

import config
from benchmarking.metrics import compute_insights
from benchmarking.system_monitor import get_system_info
from dashboard import components as C
from dashboard import visualizations as V

st.set_page_config(page_title="Sorting Performance AI", page_icon="⚡", layout="wide")
st.markdown(C.CSS, unsafe_allow_html=True)

SECTIONS = ["System Overview", "Run Experiments", "Performance Comparison", "Machine Learning",
            "Execution Time Prediction", "Dataset & Results"]
MODEL_LABELS = {"best": "Best model (selected by CV)", "linear_regression": "Linear Regression",
                "decision_tree": "Decision Tree", "random_forest": "Random Forest"}

st.sidebar.title("⚡ Sorting Performance AI")
st.sidebar.caption("Serial vs parallel sorting on multicore CPUs, with ML execution-time prediction")
section = st.sidebar.radio("Section", SECTIONS, label_visibility="collapsed")


@st.cache_data(show_spinner=False)
def system_info() -> dict:
    return get_system_info()


def fmt_time(s: float) -> str:
    return f"{s * 1000:,.2f} ms" if s < 1 else f"{s:,.3f} s"


def nan_or(v, fmt: str, missing: str = "n/a (run too short)") -> str:
    return missing if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


# =========================================================================== 1. System overview
def page_overview() -> None:
    st.title("System Overview")
    info = system_info()
    raw = C.load_raw()
    meta = C.load_model_metadata()
    c = st.columns(4)
    c[0].metric("Physical cores", info["physical_cores"])
    c[1].metric("Logical cores", info["logical_cores"])
    vm = psutil.virtual_memory()
    c[2].metric("RAM available", f"{vm.available / 1024**3:.1f} / {vm.total / 1024**3:.1f} GB")
    c[3].metric("CPU utilisation (aggregate, now)", f"{psutil.cpu_percent(interval=0.3):.1f}%")
    c = st.columns(4)
    c[0].metric("Benchmark records", f"{len(raw):,}")
    c[1].metric("Configurations", f"{raw.groupby(['algorithm', 'workers', 'input_size']).ngroups:,}" if not raw.empty else 0)
    c[2].metric("ML models", "Trained" if meta else "Not trained")
    c[3].metric("Best model", MODEL_LABELS.get(meta["best_model"], meta["best_model"]) if meta else "—")

    st.subheader("Hardware & software")
    st.table(pd.DataFrame({"Property": ["CPU", "Max frequency", "Operating system", "Python",
                                        "Multiprocessing start method"],
                           "Value": [info["cpu_model"], f"{info['cpu_max_freq_mhz']} MHz",
                                     info["os"], info["python_version"], "spawn (Windows)"]}))
    if meta:
        st.caption(f"Models trained {meta['trained_at']} on {meta['n_samples']} measurements "
                   f"({meta['n_configurations']} configurations).")

    st.subheader("Per-logical-CPU utilisation (instantaneous sample)")
    per = psutil.cpu_percent(interval=0.5, percpu=True)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.bar(range(len(per)), per, color=V.SERIES[0], width=0.65)
    ax.set(xlabel="Logical CPU", ylabel="%", ylim=(0, 100))
    ax.set_xticks(range(len(per)))
    ax.grid(axis="x", visible=False)
    C.show_fig(fig)
    if st.button("Refresh"):
        st.rerun()


# =========================================================================== 2. Run experiments
def _algo_picker(key: str) -> tuple[str, str, int]:
    mode = st.radio("Execution mode", [config.SERIAL, config.PARALLEL], horizontal=True, key=f"{key}_mode")
    algos = [a for a, m in config.ALGORITHM_MODES.items() if m == mode]
    algo = st.selectbox("Sorting algorithm", algos, format_func=config.ALGORITHM_LABELS.get, key=f"{key}_algo_{mode}")
    if mode == config.PARALLEL:
        opts = list(range(1, config.LOGICAL_CORES + 1))
        workers = st.select_slider("Worker processes", opts, value=min(4, config.LOGICAL_CORES), key=f"{key}_w")
    else:
        workers = 1
        st.caption("Serial algorithms run in a single process (workers = 1).")
    return mode, algo, workers


def page_experiments() -> None:
    st.title("Run Experiments")
    tab1, tab2 = st.tabs(["Single benchmark", "Benchmark suite"])
    with tab1:
        st.caption("Each run executes in a fresh Python process so the dashboard cannot contaminate the timing. "
                   "Output is validated against Python's sorted() before any result is shown or saved.")
        left, right = st.columns([1, 2])
        with left:
            mode, algo, workers = _algo_picker("single")
            size = st.number_input("Input size (elements)", 1, config.MAX_INPUT_SIZE, 100_000, step=10_000)
            rep = st.number_input("Dataset repetition index (seeded dataset)", 0, 1000, 0)
            seed = st.number_input("Random seed", 0, 10**9, config.DEFAULT_SEED)
            save = st.checkbox("Append result to benchmark dataset", value=False)
            run = st.button("▶ Run benchmark", type="primary")
        with right:
            if run:
                with st.spinner(f"Running {config.ALGORITHM_LABELS[algo]} on {int(size):,} elements..."):
                    try:
                        st.session_state["single_result"] = C.run_single_subprocess(
                            algo, int(size), int(workers), int(rep), int(seed), save)
                    except Exception as exc:  # surfaced to the user, not swallowed
                        st.session_state.pop("single_result", None)
                        st.error(f"Benchmark failed: {exc}")
                if save and "single_result" in st.session_state:
                    st.toast("Result appended to dataset")
            res = st.session_state.get("single_result")
            if res:
                _show_single_result(res)
            else:
                st.info("Configure a run on the left and press **Run benchmark**.")
    with tab2:
        _suite_ui()


def _show_single_result(r: dict) -> None:
    st.subheader(f"{config.ALGORITHM_LABELS[r['algorithm']]} · n = {int(r['input_size']):,} · "
                 f"{int(r['workers'])} worker(s)")
    st.success("✅ Correctness validated: output is identical to Python sorted()" if r["correct"]
               else "❌ Output incorrect")
    c = st.columns(4)
    c[0].metric("Execution time", fmt_time(r["exec_time_s"]))
    c[1].metric("System CPU (aggregate)", nan_or(r["system_cpu_percent"], "{:.1f}%"))
    c[2].metric("Effective logical CPUs used", nan_or(r["effective_cores_used"], "{:.2f}"))
    c[3].metric("Process-tree CPU time", f"{r['process_cpu_time_s']:.3f} s")
    if r["execution_mode"] == config.PARALLEL:
        c = st.columns(4)
        c[0].metric("Pool startup (excluded)", fmt_time(r["startup_time_s"]))
        c[1].metric("Scatter + sort + gather", fmt_time(r["parallel_phase_time_s"]))
        c[2].metric("k-way merge", fmt_time(r["merge_time_s"]))
        c[3].metric("Load imbalance (max/mean)", f"{r['load_imbalance']:.3f}")
        st.caption(f"Estimated IPC/pickling overhead: {fmt_time(r['communication_overhead_s'])}; "
                   f"slowest worker sorting time: {fmt_time(r['worker_compute_max_s'])}.")
    summary = C.load_summary()
    if not summary.empty:
        base = summary[(summary["algorithm"] == config.BASELINE_ALGORITHM) & (summary["input_size"] == r["input_size"])]
        same = summary[(summary["algorithm"] == r["algorithm"]) & (summary["workers"] == r["workers"])
                       & (summary["input_size"] == r["input_size"])]
        if not base.empty and r["algorithm"] in config.SPEEDUP_FAMILY:
            sp = base["median_time_s"].iloc[0] / r["exec_time_s"]
            st.write(f"Speedup vs historical serial merge sort median at this size: **{sp:.2f}x**"
                     + (f" (efficiency {sp / r['workers']:.0%})." if r["execution_mode"] == config.PARALLEL else "."))
        if not same.empty:
            st.write(f"Historical median for this exact configuration: **{fmt_time(same['median_time_s'].iloc[0])}** "
                     f"over {int(same['n_runs'].iloc[0])} runs.")
    st.caption("CPU utilisation is aggregate (all logical CPUs) or process-tree based — not per-core. "
               "Runs under 50 ms are too short for Windows CPU accounting and show n/a.")


def _suite_ui() -> None:
    st.caption("Runs the full experiment grid in a background Python process and appends results to the dataset. "
               "Large sizes take minutes (pure-Python merge sort of 500k elements ≈ 2 s per run).")
    c1, c2 = st.columns(2)
    sizes_txt = c1.text_input("Input sizes (comma-separated)", ", ".join(map(str, config.DEFAULT_SIZES)))
    workers = c1.multiselect("Worker counts", list(range(1, config.LOGICAL_CORES + 1)), config.DEFAULT_WORKER_COUNTS)
    reps = c2.slider("Repetitions per configuration", 1, 10, config.DEFAULT_REPETITIONS)
    seed = c2.number_input("Random seed", 0, 10**9, config.DEFAULT_SEED, key="suite_seed")
    algos = st.multiselect("Algorithms", list(config.ALGORITHM_MODES), list(config.ALGORITHM_MODES),
                           format_func=config.ALGORITHM_LABELS.get)
    if st.button("▶ Run benchmark suite", type="primary"):
        try:
            sizes = C.parse_int_list(sizes_txt)
            if any(not 1 <= s <= config.MAX_INPUT_SIZE for s in sizes):
                raise ValueError(f"sizes must be between 1 and {config.MAX_INPUT_SIZE:,}")
            if not algos:
                raise ValueError("select at least one algorithm")
            if "parallel_merge_sort" in algos and not workers:
                raise ValueError("select at least one worker count")
        except ValueError as exc:
            st.error(f"Invalid settings: {exc}")
            return
        bar = st.progress(0.0, text="Starting...")
        log = st.empty()
        lines: list[str] = []

        def on_line(line: str) -> None:
            lines.append(line)
            if line.startswith("[") and "%]" in line:
                try:
                    bar.progress(min(1.0, float(line[1:line.index("%")]) / 100), text=line.split("] ", 1)[-1])
                except ValueError:
                    pass
            log.code("\n".join(lines[-12:]))

        code = C.run_suite_subprocess(sizes, workers or [1], int(reps), int(seed), algos, on_line)
        if code == 0:
            bar.progress(1.0, text="Complete")
            C.write_analysis_outputs()
            st.cache_data.clear()
            st.success("Benchmark suite finished; dataset, summary and plots updated.")
        else:
            st.error(f"Benchmark process failed (exit code {code}). See log above.")


# =========================================================================== 3. Performance comparison
def page_performance() -> None:
    st.title("Performance Comparison")
    summary = C.load_summary()
    if summary.empty:
        st.warning("No benchmark data yet. Run experiments first.")
        return
    sizes = sorted(summary["input_size"].unique().astype(int))
    n = st.select_slider("Input size for per-size charts", sizes, value=sizes[-1], format_func=lambda v: f"{v:,}")
    t1, t2, t3, t4, t5 = st.tabs(["Serial vs parallel", "By input size", "Speedup & efficiency",
                                  "Overheads", "CPU utilisation"])
    with t1:
        C.show_fig(V.plot_serial_vs_parallel(summary, n))
    with t2:
        C.show_fig(V.plot_time_vs_size(summary))
    with t3:
        a, b = st.columns(2)
        with a:
            C.show_fig(V.plot_speedup_vs_workers(summary))
        with b:
            C.show_fig(V.plot_efficiency_vs_workers(summary))
    with t4:
        C.show_fig(V.plot_overhead_breakdown(summary, n))
    with t5:
        C.show_fig(V.plot_cpu_utilization(summary, n))
        st.caption("Left: CPU seconds of the benchmark process + workers ÷ wall time. Right: machine-wide "
                   "aggregate utilisation (psutil), including background load. Neither is a per-core figure.")

    st.subheader("Analysis (computed from the measurements)")
    for f in compute_insights(summary)["findings"]:
        st.markdown(f"- {f}")

    st.subheader("Summary statistics")
    cols = ["algorithm", "input_size", "workers", "n_runs", "mean_time_s", "median_time_s", "min_time_s",
            "std_time_s", "cv_percent", "speedup", "efficiency", "speedup_vs_timsort", "mean_startup_time_s",
            "end_to_end_speedup", "mean_effective_cores_used", "mean_system_cpu_percent"]
    st.dataframe(summary[cols].round(5), hide_index=True)
    if st.button("Save summary CSV, insights and plots to disk"):
        paths = C.write_analysis_outputs()
        st.success(f"Saved {len(paths)} plots to {config.PLOTS_DIR} and summary to {config.SUMMARY_CSV}")


# =========================================================================== 4. Machine learning
def page_ml() -> None:
    st.title("Machine Learning")
    raw = C.load_raw()
    st.caption("Models: Linear Regression, Decision Tree, Random Forest. Target: execution time (s), learned on a "
               "log scale. Repetitions of a configuration (algorithm, workers, size) are grouped so they never "
               "straddle train/test. Selection: GroupKFold CV RMSE; the winner is then scored on held-out configurations.")
    c = st.columns([1, 1, 2])
    test_size = c[0].slider("Held-out test fraction (configurations)", 0.15, 0.4, 0.25, 0.05)
    c[1].metric("Training measurements", f"{len(raw):,}")
    if c[2].button("🧠 Train all models", type="primary", disabled=raw.empty):
        from ml.preprocessing import InsufficientDataError
        from ml.train import train_models
        with st.status("Training models...", expanded=True) as status:
            try:
                res = train_models(raw, test_size=test_size, progress=st.write)
                status.update(label=f"Training complete — best model: {MODEL_LABELS[res.best_model]}", state="complete")
                st.cache_data.clear()
            except InsufficientDataError as exc:
                status.update(label="Not enough data", state="error")
                st.error(str(exc))
    if raw.empty:
        st.warning("No benchmark data yet — run experiments first.")

    metrics = C.load_csv(config.MODEL_METRICS_CSV)
    meta = C.load_model_metadata()
    if metrics.empty or not meta:
        st.info("Models have not been trained yet.")
        return
    st.success(f"Best model by cross-validated RMSE: **{MODEL_LABELS[meta['best_model']]}** "
               f"(trained {meta['trained_at']}, {meta['n_train_samples']} train / {meta['n_test_samples']} test rows)")
    st.subheader("Model comparison")
    view = metrics.copy()
    view["model"] = view["model"].map(lambda m: MODEL_LABELS.get(m, m))
    st.dataframe(view[["model", "split", "MAE", "MSE", "RMSE", "R2", "MAPE"]].round(6), hide_index=True)
    metric = st.radio("Metric", ["RMSE", "MAE", "MSE", "R2", "MAPE"], horizontal=True)
    C.show_fig(V.plot_model_comparison(metrics, metric))
    ext = meta.get("extrapolation_check")
    if ext:
        st.warning(f"Extrapolation check: {MODEL_LABELS[meta['best_model']]} retrained on sizes ≤ "
                   f"{ext['trained_up_to']:,} and tested on unseen n = {ext['held_out_input_size']:,}: "
                   f"R² = {ext['R2']:.3f}, MAPE = {ext['MAPE']:.1%}. Predictions outside the benchmarked size "
                   "range are much less reliable than interpolation; the model only knows this machine.")

    st.subheader("Actual vs predicted (held-out test configurations)")
    preds = C.load_csv(config.MODEL_PREDICTIONS_CSV)
    imps = C.load_csv(config.FEATURE_IMPORTANCE_CSV)
    model = st.selectbox("Model", ["linear_regression", "decision_tree", "random_forest"], format_func=MODEL_LABELS.get,
                         index=["linear_regression", "decision_tree", "random_forest"].index(meta["best_model"]))
    test_m = metrics[(metrics["split"] == "test") & (metrics["model"] == model)]
    a, b = st.columns(2)
    with a:
        C.show_fig(V.plot_actual_vs_predicted(preds, model, MODEL_LABELS[model],
                                              float(test_m["R2"].iloc[0]) if not test_m.empty else None))
    with b:
        C.show_fig(V.plot_feature_importance(imps[imps["model"] == model], MODEL_LABELS[model]))
    st.subheader("Saved model files")
    files = sorted(config.MODELS_DIR.glob("*.joblib"))
    st.dataframe(pd.DataFrame({"file": [str(p) for p in files], "size_kb": [round(p.stat().st_size / 1024, 1) for p in files]}),
                 hide_index=True)


# =========================================================================== 5. Prediction
def page_predict() -> None:
    st.title("Execution Time Prediction")
    meta = C.load_model_metadata()
    if not meta:
        st.warning("No trained model found. Train the models in the Machine Learning section first.")
        return
    left, right = st.columns([1, 2])
    with left:
        mode, algo, workers = _algo_picker("pred")
        size = st.number_input("Input size (elements)", 1, config.MAX_INPUT_SIZE, 200_000, step=10_000, key="pred_n")
        model = st.selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
        go = st.button("🔮 Predict", type="primary")
    with right:
        if go:
            from ml.predict import predict_execution_time
            try:
                st.session_state["prediction"] = (predict_execution_time(int(size), algo, mode, int(workers), model),
                                                  int(size), algo, int(workers))
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                st.session_state.pop("prediction", None)
                st.error(str(exc))
        if "prediction" not in st.session_state:
            st.info("Choose a configuration and press **Predict**.")
            return
        p, n, a, w = st.session_state["prediction"]
        st.subheader(f"{config.ALGORITHM_LABELS[a]} · n = {n:,} · {w} worker(s)")
        c = st.columns(3)
        c[0].metric("Predicted time (s)", f"{p.predicted_s:.6f}")
        c[1].metric("Predicted time (ms)", f"{p.predicted_ms:,.3f}")
        c[2].metric("Model", MODEL_LABELS.get(p.model_name, p.model_name))
        if p.exact_match:
            st.info(f"Measured median for this exact configuration: {fmt_time(p.exact_match['median_time_s'])} "
                    f"(n_runs = {int(p.exact_match['n_runs'])}); prediction error "
                    f"{p.exact_match['prediction_error_pct']:+.2f}%.")
        for warn in p.warnings:
            st.warning(warn)
        if not p.history.empty:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.plot(p.history["input_size"], p.history["median_time_s"], marker="o", color=V.SERIES[0],
                    label="Measured median (benchmarks)")
            ax.scatter([n], [p.predicted_s], s=90, marker="D", color=V.SERIES[1], zorder=3,
                       edgecolor=V.SURFACE, linewidth=1.5, label="Prediction")
            ax.set(xscale="log", yscale="log", xlabel="Input size", ylabel="Time (s)",
                   title="Prediction vs historical measurements")
            ax.legend(fontsize=8)
            C.show_fig(fig)
            st.dataframe(p.history.round(6), hide_index=True)
        else:
            st.caption("No historical measurements for this algorithm/worker combination.")


# =========================================================================== 6. Dataset & results
def page_data() -> None:
    st.title("Dataset & Results")
    raw = C.load_raw()
    if raw.empty:
        st.warning("No benchmark records yet.")
    else:
        c = st.columns(3)
        algos = c[0].multiselect("Algorithm", sorted(raw["algorithm"].unique()))
        sizes = c[1].multiselect("Input size", sorted(raw["input_size"].unique()))
        workers = c[2].multiselect("Workers", sorted(raw["workers"].unique()))
        f = raw
        if algos:
            f = f[f["algorithm"].isin(algos)]
        if sizes:
            f = f[f["input_size"].isin(sizes)]
        if workers:
            f = f[f["workers"].isin(workers)]
        st.caption(f"{len(f):,} of {len(raw):,} records")
        st.dataframe(f, hide_index=True, height=380)
        st.download_button("⬇ Download filtered records (CSV)", f.to_csv(index=False).encode(),
                           "benchmark_results_filtered.csv", "text/csv")
    st.subheader("Downloads")
    c = st.columns(3)
    with c[0]:
        C.download_file_button("⬇ Raw benchmark dataset", config.RAW_BENCHMARK_CSV)
        if not raw.empty:
            st.download_button("⬇ Summary statistics", C.load_summary().to_csv(index=False).encode(),
                               "benchmark_summary.csv", "text/csv")
    with c[1]:
        C.download_file_button("⬇ Model evaluation metrics", config.MODEL_METRICS_CSV)
        C.download_file_button("⬇ Test-set predictions", config.MODEL_PREDICTIONS_CSV)
        C.download_file_button("⬇ Feature importance", config.FEATURE_IMPORTANCE_CSV)
        C.download_file_button("⬇ Model metadata (JSON)", config.MODEL_METADATA_JSON, "application/json")
    with c[2]:
        plots = sorted(config.PLOTS_DIR.glob("*.png"))
        if plots:
            st.download_button(f"⬇ All graphs (ZIP, {len(plots)} files)", C.plots_zip(), "plots.zip", "application/zip")
            choice = st.selectbox("Single graph", plots, format_func=lambda p: p.name)
            C.download_file_button("⬇ Download selected graph", choice, "image/png", key="one_plot")
            st.image(str(choice))
        else:
            st.caption("No graphs saved yet. Use 'Save summary CSV, insights and plots' or train the models.")


PAGES = dict(zip(SECTIONS, [page_overview, page_experiments, page_performance, page_ml, page_predict, page_data]))
PAGES[section]()
