import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# =====================================================================
# 1. WEIGHT QUANTIZATION SIMULATION (FP32 to INT8)
# =====================================================================
def quantize_weights(weights: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Quantize floating point weights to 8-bit integers (asymmetric)."""
    w_min, w_max = weights.min(), weights.max()
    
    # Calculate scale and zero-point
    # We map [w_min, w_max] to [0, 255] range for uint8
    scale = (w_max - w_min) / 255.0
    if scale == 0:
        scale = 1.0
    zero_point = round((0.0 - w_min) / scale)
    # Clip zero point to uint8 boundaries
    zero_point = max(0, min(255, zero_point))
    
    # Quantize
    q_weights = np.round(weights / scale) + zero_point
    q_weights = np.clip(q_weights, 0, 255).astype(np.uint8)
    
    return q_weights, scale, zero_point

def dequantize_weights(q_weights: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Restore quantized integers back to floating point representation."""
    return (q_weights.astype(np.float32) - zero_point) * scale

def run_quantization_experiment():
    print("\n--- Running FP32 to INT8 Quantization Simulator ---")
    # Simulate a layer weight matrix (e.g. 500x500 weights)
    np.random.seed(42)
    original_weights = np.random.randn(500, 500).astype(np.float32) * 0.5
    
    q_weights, scale, zero_point = quantize_weights(original_weights)
    dequantized_weights = dequantize_weights(q_weights, scale, zero_point)
    
    # Calculate size and reconstruction error
    original_size = original_weights.nbytes
    quantized_size = q_weights.nbytes
    compression_ratio = original_size / quantized_size
    
    mse = mean_squared_error(original_weights.flatten(), dequantized_weights.flatten())
    
    print(f"Original size (FP32):   {original_size} bytes")
    print(f"Quantized size (INT8):  {quantized_size} bytes")
    print(f"Compression Ratio:      {compression_ratio:.1f}x (75% savings)")
    print(f"Reconstruction MSE:     {mse:.6f}")
    
    return {
        "original_size": original_size,
        "quantized_size": quantized_size,
        "compression_ratio": compression_ratio,
        "reconstruction_mse": mse
    }

# =====================================================================
# 2. KNOWLEDGE DISTILLATION (Teacher to Student)
# =====================================================================
def run_distillation_experiment():
    print("\n--- Running Knowledge Distillation Experiment ---")
    
    # Generate synthetic regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=5.0, random_state=42)
    
    # Split into train/test
    split = 800
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 1. Train Teacher Model (Complex Ensemble)
    teacher = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    teacher.fit(X_train, y_train)
    teacher_preds = teacher.predict(X_test)
    teacher_mse = mean_squared_error(y_test, teacher_preds)
    
    # Generate soft targets (Teacher's predictions on training data)
    soft_targets = teacher.predict(X_train)
    
    # 2. Train Base Student Model directly on Hard Labels (Shallow Decision Tree)
    student_hard = DecisionTreeRegressor(max_depth=3, random_state=42)
    student_hard.fit(X_train, y_train)
    student_hard_preds = student_hard.predict(X_test)
    student_hard_mse = mean_squared_error(y_test, student_hard_preds)
    
    # 3. Train Distilled Student Model on Soft Labels (Teacher's outputs)
    student_soft = DecisionTreeRegressor(max_depth=3, random_state=42)
    student_soft.fit(X_train, soft_targets)
    student_soft_preds = student_soft.predict(X_test)
    student_soft_mse = mean_squared_error(y_test, student_soft_preds)
    
    print(f"Teacher Model (RF) MSE:                    {teacher_mse:.4f}")
    print(f"Student Model (Tree, Hard Labels) MSE:     {student_hard_mse:.4f}")
    print(f"Student Model (Tree, Distilled Soft) MSE:  {student_soft_mse:.4f}")
    
    # Measure sizes (number of nodes in decision trees as size metric)
    teacher_node_count = sum(tree.tree_.node_count for tree in teacher.estimators_)
    student_hard_nodes = student_hard.tree_.node_count
    student_soft_nodes = student_soft.tree_.node_count
    
    print(f"Teacher Model complexity (total nodes):    {teacher_node_count}")
    print(f"Student Model complexity (nodes):          {student_hard_nodes}")
    
    return {
        "teacher_mse": teacher_mse,
        "student_hard_mse": student_hard_mse,
        "student_soft_mse": student_soft_mse,
        "teacher_nodes": teacher_node_count,
        "student_nodes": student_hard_nodes
    }

# =====================================================================
# MAIN RUNNER
# =====================================================================
def main():
    print("="*60)
    print("EXPERIMENT 11: MODEL OPTIMIZATION EXPERIMENT START")
    print("="*60)
    
    quant_res = run_quantization_experiment()
    dist_res = run_distillation_experiment()
    
    out_dir = os.path.dirname(__file__)
    
    # Save Report
    with open(os.path.join(out_dir, "optimization_report.json"), "w") as f:
        json.dump({
            "quantization": quant_res,
            "distillation": dist_res
        }, f, indent=4)
        
    # Plot results
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Quantization Size Comparison
        sizes = [quant_res["original_size"] / 1024, quant_res["quantized_size"] / 1024]
        labels = ["Original FP32", "Quantized INT8"]
        ax1.bar(labels, sizes, color=["blue", "green"], width=0.5, alpha=0.7)
        ax1.set_ylabel("Memory Footprint (KB)")
        ax1.set_title("Post-Training Quantization Comparison")
        for i, v in enumerate(sizes):
            ax1.text(i, v + 2, f"{v:.1f} KB", ha='center', va='bottom', fontweight='bold')
            
        # Plot 2: Distillation MSE Comparison
        mses = [dist_res["teacher_mse"], dist_res["student_hard_mse"], dist_res["student_soft_mse"]]
        models = ["Teacher (RF)", "Student (Hard Labels)", "Student (Distilled Soft)"]
        ax2.bar(models, mses, color=["purple", "red", "orange"], width=0.5, alpha=0.7)
        ax2.set_ylabel("Mean Squared Error (MSE)")
        ax2.set_title("Knowledge Distillation Performance")
        for i, v in enumerate(mses):
            ax2.text(i, v + 5, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        chart_path = os.path.join(out_dir, "optimization_chart.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"\n[Optimization] Charts saved successfully to {chart_path}")
    except Exception as e:
        print(f"[Optimization] Failed to create chart visualization: {e}")
        
    print("="*60)

if __name__ == "__main__":
    main()
