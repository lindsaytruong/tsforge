from dataclasses import dataclass, field
from typing import Callable, List
import pandas as pd

Step = Callable[[pd.DataFrame], pd.DataFrame]

def step_as_category(cols: List[str]) -> Step:
    def _fn(df: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype("category")
        return df
    return _fn

@dataclass
class Recipe:
    name: str = "recipe"
    steps: List[Step] = field(default_factory=list)

    def add_step(self, fn: Step) -> "Recipe":
        self.steps.append(fn)
        return self

    def step_as_category(self, cols: List[str]):
        return self.add_step(step_as_category(cols))

    def bake(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for fn in self.steps:
            out = fn(out)
        return out
