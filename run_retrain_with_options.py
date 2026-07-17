"""
run_retrain_with_options.py
量化策略重训主调度脚本。
依次执行：
1. 增量更新期权数据 (data/update_options_data.py)
2. 重新构建包含期权特征的 Parquet 矩阵 (research/期权/build_features_with_options.py)
3. 滚动重训 XGBoost 模型并生成最新预测 (research/期权/train_models_with_options.py)
"""
import sys
import os
import time

def run_retrain():
    t0 = time.time()
    print("==========================================================")
    print("   Starting Quant Model Retrain with Option Features      ")
    print("==========================================================")

    # Add workspace path to sys.path
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)

    # 1. Update Options PCR Data
    print("\n[STEP 1/3] Syncing latest options PCR data...")
    try:
        from data.update_options_data import main as sync_options
        sync_options()
        print("[SUCCESS] Options PCR data synced.")
    except Exception as e:
        print(f"[ERROR] Step 1 failed: {e}")
        return False

    # 2. Build Features with Options
    print("\n[STEP 2/3] Rebuilding option feature matrix...")
    try:
        # Change current directory to locate files correctly if needed, or import
        from research.期权.build_features_with_options import main as build_features
        build_features()
        print("[SUCCESS] Features successfully built.")
    except Exception as e:
        print(f"[ERROR] Step 2 failed: {e}")
        return False

    # 3. Walk-Forward XGBoost Retrain
    print("\n[STEP 3/3] Running Walk-Forward training on XGBoost models...")
    try:
        from research.期权.train_models_with_options import run as train_models
        train_models()
        print("[SUCCESS] Walk-Forward model training completed.")
    except Exception as e:
        print(f"[ERROR] Step 3 failed: {e}")
        return False

    elapsed = time.time() - t0
    print("\n==========================================================")
    print(f"   Model Retrain Complete! Time elapsed: {elapsed/60:.1f} minutes")
    print("==========================================================")
    return True

if __name__ == "__main__":
    success = run_retrain()
    sys.exit(0 if success else 1)
