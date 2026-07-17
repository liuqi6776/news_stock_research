import joblib
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for name in ['daily_dragon_news_model.joblib', 'daily_dragon_model.joblib']:
    path = os.path.join(ROOT_DIR, name)
    if os.path.exists(path):
        try:
            model_data = joblib.load(path)
            # Some joblibs contain (model, features), others are dicts
            if isinstance(model_data, tuple):
                model, feats = model_data
            elif isinstance(model_data, dict):
                feats = model_data.get('features', [])
            else:
                feats = getattr(model_data, 'feature_names_in_', [])
                
            print(f"Model: {name}")
            print(f"  Features count: {len(feats)}")
            print(f"  Features list: {list(feats)}")
            # Check for option-related terms
            opt_feats = [f for f in feats if 'opt' in str(f) or 'pcr' in str(f) or 'vix' in str(f)]
            if opt_feats:
                print(f"  [FOUND OPTIONS FEATURES]: {opt_feats}")
            else:
                print("  [NO OPTIONS FEATURES FOUND]")
            print("-" * 50)
        except Exception as e:
            print(f"Error loading {name}: {e}")
    else:
        print(f"Model file {name} does not exist!")
