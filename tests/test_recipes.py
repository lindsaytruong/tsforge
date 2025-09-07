import pandas as pd
from tsforge.feature_engineering.recipes import Recipe

def test_step_as_category():
    df = pd.DataFrame({"id": [1,2,3]})
    recipe = Recipe().step_as_category(["id"])
    result = recipe.bake(df)
    assert str(result["id"].dtype) == "category"
