import joblib
import pandas as pd
path = r'c:\Users\liuqi\quant_system_v2\new_idea\final_result\doubao\models\doubao_t1t2_model.joblib'
loaded = joblib.load(path)
print(f'Type: {type(loaded)}')
if isinstance(loaded, tuple):
    print(f'Len: {len(loaded)}')
    for i, item in enumerate(loaded):
        print(f'  [{i}] type={type(item)}, classes={getattr(item, "classes_", None)}')
    model = loaded[0]
    feats = loaded[1] if len(loaded) > 1 else None
    print(f'Model classes: {getattr(model, "classes_", None)}')
    print(f'Feats: {feats}')
    X_test = pd.DataFrame({'hot_rank_pct': [0.5], 'chip_concentration': [0.3], 'winner_rate': [0.5], 'news_market_impact': [0], 'news_stock_impact': [0]})
    pred = model.predict_proba(X_test)
    print(f'Predict proba shape: {pred.shape}')
    print(f'Predict proba: {pred}')
else:
    print(f'Classes: {getattr(loaded, "classes_", None)}')
    X_test = pd.DataFrame({'hot_rank_pct': [0.5], 'chip_concentration': [0.3], 'winner_rate': [0.5], 'news_market_impact': [0], 'news_stock_impact': [0]})
    pred = loaded.predict_proba(X_test)
    print(f'Predict proba shape: {pred.shape}')
    print(f'Predict proba: {pred}')
