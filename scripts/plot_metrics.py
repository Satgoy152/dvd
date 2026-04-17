import json
import os
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_results(results_dir):
    data = []
    search_path = os.path.join(results_dir, "*.json")
    for fp in glob.glob(search_path):
        with open(fp, "r") as f:
            try:
                run_data = json.load(f)
            except Exception as e:
                print(f"Skipping {fp}: parse error - {e}")
                continue
                
        config = run_data.get("config", {})
        metrics = run_data.get("dvd_metrics_average", {})
        scores = run_data.get("task_scores", {})
        
        algorithm = config.get("algorithm", "unknown")
        steps = config.get("steps", 0)
        
        # Determine aggregate accuracy
        valid_scores = [v for v in scores.values() if v is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        # NFEs
        v_nfe = metrics.get("verifier_nfe", 0)
        d_nfe = metrics.get("drafter_nfe", 0)
        total_nfe = v_nfe + d_nfe
        
        # Latency
        latency = metrics.get("total_latency", 0)
        
        data.append({
            "algorithm": algorithm,
            "steps": steps,
            "accuracy": avg_score,
            "total_nfe": total_nfe,
            "verifier_nfe": v_nfe,
            "drafter_nfe": d_nfe,
            "latency": latency,
            "filename": os.path.basename(fp)
        })
    return data

def plot_pareto(data_points, x_col, x_label, title, out_path):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Group by algorithm for lines
    algorithms = set(d["algorithm"] for d in data_points)
    
    colors = sns.color_palette("husl", len(algorithms))
    
    for i, algo in enumerate(algorithms):
        pts = [d for d in data_points if d["algorithm"] == algo]
        # Sort by x_col to plot a coherent line
        pts.sort(key=lambda d: d[x_col])
        
        x_vals = [d[x_col] for d in pts]
        y_vals = [d["accuracy"] for d in pts]
        
        plt.plot(x_vals, y_vals, marker='o', label=algo, color=colors[i], linewidth=2, markersize=8)
        
        for pt in pts:
            plt.text(pt[x_col], pt["accuracy"], f" {pt['steps']} steps", fontsize=9, alpha=0.7)
            
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(title="Algorithm")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Pareto Frontiers from Evaluation Results")
    parser.add_argument("--results_dir", type=str, default="eval_results", help="Directory containing JSON results")
    parser.add_argument("--output_dir", type=str, default="eval_results/plots", help="Directory to save generated plots")
    args = parser.parse_args()

    data = parse_results(args.results_dir)
    
    if not data:
        print(f"No JSON data found in {args.results_dir}.")
        return
        
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot Accuracy vs Total NFE
    plot_pareto(
        data, 
        "total_nfe", 
        "Total Core Forward Passes (Drafter + Verifier)",
        "Pareto Frontier: Accuracy vs System NFEs", 
        out_dir / "pareto_nfe_vs_acc.png"
    )
    
    # 2. Plot Accuracy vs Verification Steps
    plot_pareto(
        data, 
        "steps", 
        "Outer Generation Steps", 
        "Pareto Frontier: Accuracy vs Outer Generation Steps", 
        out_dir / "pareto_steps_vs_acc.png"
    )
    
    # 3. Plot Accuracy vs Latency
    plot_pareto(
        data, 
        "latency", 
        "Total Latency (Seconds)", 
        "Pareto Frontier: Accuracy vs Total System Latency", 
        out_dir / "pareto_latency_vs_acc.png"
    )

if __name__ == "__main__":
    main()
