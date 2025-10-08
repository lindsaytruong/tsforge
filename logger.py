# src/tsforge/logging/logger.py

import time
import pandas as pd
from typing import Dict, Callable, Optional, List
from datetime import datetime

class WorkflowLogger:
    """Simple logger for tracking workflow execution."""
    
    def __init__(self, verbose: int = 1):
        """
        Initialize logger.
        
        Parameters
        ----------
        verbose : int
            0 = silent, 1 = basic, 2 = detailed
        """
        self.verbose = verbose
        self._execution_times = {}
        self._workflow_info = {}
        self._start_times = {}
    
    # ========================================
    # Recipe Methods
    # ========================================
    
    def log_recipe_start(self, recipe_name: str, num_steps: int, input_shape: tuple):
        """Log recipe start."""
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Recipe: {recipe_name}")
            print(f"Steps: {num_steps}")
            print(f"Input shape: {input_shape}")
            print(f"{'='*60}")
        self._start_times[f"recipe_{recipe_name}"] = time.time()
    
    def log_recipe_step(self, step_num: int, step_name: str, before_shape: tuple, after_shape: tuple):
        """Log individual recipe step."""
        if self.verbose >= 2:
            row_diff = after_shape[0] - before_shape[0]
            col_diff = after_shape[1] - before_shape[1]
            print(f"  [{step_num}] {step_name}")
            print(f"      Shape: {before_shape} → {after_shape}", end="")
            if row_diff != 0 or col_diff != 0:
                print(f" ({row_diff:+d} rows, {col_diff:+d} cols)", end="")
            print()
    
    def log_recipe_complete(self, recipe_name: str, final_shape: tuple):
        """Log recipe completion."""
        key = f"recipe_{recipe_name}"
        if key in self._start_times:
            elapsed = time.time() - self._start_times[key]
            self._execution_times[recipe_name] = elapsed
            if self.verbose >= 1:
                print(f"Final shape: {final_shape}")
                print(f"✓ Recipe completed in {elapsed:.2f}s")
                print(f"{'='*60}\n")
    
    # ========================================
    # Workflow Methods
    # ========================================
    
    def init(self, workflows: List[dict]):
        """Initialize with workflow list."""
        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print(f"🚀 WorkflowManager initialized with {len(workflows)} workflow(s)")
            for i, wf in enumerate(workflows, 1):
                engine = wf['engine']
                name = wf['name']
                has_recipe = '✓' if wf.get('recipe') else '✗'
                num_models = len(wf['params'].get('models', []))
                print(f"   [{i}] {name:20s} | Engine: {engine:12s} | Models: {num_models:2d} | Recipe: {has_recipe}")
            print(f"{'='*80}\n")
    
    def workflow_start(self, name: str, num_series: int, num_rows: int):
        """Log workflow start."""
        if self.verbose >= 2:
            print(f"\n{'─'*60}")
            print(f"Workflow: {name}")
            print(f"Series: {num_series} | Rows: {num_rows}")
        self._start_times[f"workflow_{name}"] = time.time()
    
    def training_start(self, name: str, engine: str, num_models: int):
        """Log training start."""
        if self.verbose >= 2:
            print(f"Training with {engine}...", end="", flush=True)
    
    def training_complete(self, name: str):
        """Log training completion."""
        if self.verbose >= 2:
            print(" ✓")
    
    def workflow_complete(self, name: str):
        """Log workflow completion."""
        key = f"workflow_{name}"
        if key in self._start_times:
            elapsed = time.time() - self._start_times[key]
            self._execution_times[name] = elapsed
            if self.verbose >= 2:
                print(f"✓ {name} completed in {elapsed:.2f}s")
                print(f"{'─'*60}")
    
    def cv_start(self, h: int, n_windows: int, step_size: int, num_workflows: int, level: Optional[List[int]]):
        """Log cross-validation start."""
        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print(f"🔬 STARTING CROSS-VALIDATION")
            print(f"   Horizon: {h} steps")
            print(f"   Windows: {n_windows}")
            print(f"   Step size: {step_size}")
            print(f"   Workflows: {num_workflows}")
            if level:
                print(f"   Prediction intervals: {level}")
            print(f"{'='*80}")
        self._start_times['cv'] = time.time()
    
    def cv_complete(self, num_preds: int, num_models: int, num_workflows: int):
        """Log cross-validation completion."""
        if 'cv' in self._start_times:
            elapsed = time.time() - self._start_times['cv']
            self._execution_times['cross_validation'] = elapsed
            if self.verbose >= 1:
                print(f"\n{'='*80}")
                print(f"✓ CROSS-VALIDATION COMPLETE")
                print(f"   Predictions: {num_preds:,}")
                print(f"   Models: {num_models}")
                print(f"   Workflows: {num_workflows}")
                print(f"   Time: {elapsed:.2f}s")
                print(f"{'='*80}\n")
    
    def forecast_start(self, h: int, num_workflows: int, level: Optional[List[int]]):
        """Log forecast start."""
        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print(f"🔮 STARTING FORECAST")
            print(f"   Horizon: {h} steps")
            print(f"   Workflows: {num_workflows}")
            if level:
                print(f"   Prediction intervals: {level}")
            print(f"{'='*80}")
        self._start_times['forecast'] = time.time()
    
    def forecast_complete(self, num_preds: int, num_models: int):
        """Log forecast completion."""
        if 'forecast' in self._start_times:
            elapsed = time.time() - self._start_times['forecast']
            self._execution_times['forecast'] = elapsed
            if self.verbose >= 1:
                print(f"\n{'='*80}")
                print(f"✓ FORECAST COMPLETE")
                print(f"   Predictions: {num_preds:,}")
                print(f"   Models: {num_models}")
                print(f"   Time: {elapsed:.2f}s")
                print(f"{'='*80}\n")
    
    def ensemble_start(self, name: str, method: str, num_members: int):
        """Log ensemble start."""
        if self.verbose >= 2:
            print(f"\n  Ensemble '{name}' ({method}) with {num_members} members")
    
    def ensemble_member(self, member_name: str, idx: int, total: int):
        """Log ensemble member processing."""
        if self.verbose >= 2:
            print(f"    [{idx}/{total}] Processing {member_name}...", end="", flush=True)
    
    def ensemble_complete(self):
        """Log ensemble completion."""
        if self.verbose >= 2:
            print(" ✓")
    
    def warning(self, name: str, message: str):
        """Log warning."""
        if self.verbose >= 1:
            print(f"⚠️  {name}: {message}")
    
    def recipe_apply(self, workflow_name: str, recipe_name: str):
        """Log recipe application."""
        if self.verbose >= 2:
            print(f"  Applying recipe '{recipe_name}'...")
    
    def recipe_complete(self, workflow_name: str):
        """Log recipe application completion."""
        if self.verbose >= 2:
            print(f"  Recipe applied ✓")
    
    # ========================================
    # Summary Methods
    # ========================================
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get execution time summary as DataFrame."""
        if not self._execution_times:
            return pd.DataFrame(columns=['workflow', 'execution_time_s'])
        
        df = pd.DataFrame([
            {'workflow': name, 'execution_time_s': t}
            for name, t in self._execution_times.items()
        ])
        return df.sort_values('execution_time_s', ascending=False)


# ========================================
# Helper Functions
# ========================================

def get_recipe_callbacks(logger: WorkflowLogger) -> Dict[str, Callable]:
    """
    Create callbacks for Recipe class.
    
    Parameters
    ----------
    logger : WorkflowLogger
        The logger instance to use
    
    Returns
    -------
    dict
        Dictionary of callback functions
    
    Examples
    --------
    >>> logger = WorkflowLogger(verbose=2)
    >>> callbacks = get_recipe_callbacks(logger)
    >>> recipe = Recipe(name="My Recipe", callbacks=callbacks)
    """
    return {
        'on_start': logger.log_recipe_start,
        'on_step': logger.log_recipe_step,
        'on_complete': logger.log_recipe_complete
    }


def get_manager_callbacks(logger: WorkflowLogger) -> Dict[str, Callable]:
    """
    Create callbacks for WorkflowManager.
    
    Parameters
    ----------
    logger : WorkflowLogger
        The logger instance to use
    
    Returns
    -------
    dict
        Dictionary of callback functions
    
    Examples
    --------
    >>> logger = WorkflowLogger(verbose=2)
    >>> callbacks = get_manager_callbacks(logger)
    >>> manager = WorkflowManager(workflows, callbacks=callbacks)
    """
    return {
        'on_init': logger.init,
        'on_cv_start': logger.cv_start,
        'on_cv_complete': logger.cv_complete,
        'on_forecast_start': logger.forecast_start,
        'on_forecast_complete': logger.forecast_complete,
        'on_workflow_start': logger.workflow_start,
        'on_workflow_complete': logger.workflow_complete,
        'on_training_start': logger.training_start,
        'on_training_complete': logger.training_complete,
        'on_ensemble_start': logger.ensemble_start,
        'on_ensemble_member': logger.ensemble_member,
        'on_ensemble_complete': logger.ensemble_complete,
        'on_recipe_apply': logger.recipe_apply,
        'on_recipe_complete': logger.recipe_complete,
        'on_warning': logger.warning,
    }