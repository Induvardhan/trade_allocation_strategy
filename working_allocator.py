"""
Simplified production-ready trade allocator.
"""

import numpy as np
from scipy.optimize import minimize
import cvxpy as cp
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SolverType(Enum):
    SCIPY = "scipy"
    CVXPY = "cvxpy"

@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    default_solver: SolverType = SolverType.CVXPY
    max_iterations: int = 1000
    tolerance: float = 1e-8

class TradeAllocator:
    """Production-ready trade allocator."""
    
    def __init__(self, impact_model, total_shares, n_intervals=50, config=None):
        self.impact_model = impact_model
        self.total_shares = float(total_shares)
        self.n_intervals = int(n_intervals)
        self.config = config or OptimizerConfig()
        self.optimal_allocation = None
        logger.info(f"Initialized TradeAllocator: {self.total_shares:,.0f} shares over {self.n_intervals} intervals")
    
    def objective_function(self, x):
        """Calculate total impact."""
        try:
            x = np.maximum(0, x)  # Ensure non-negative
            impacts = self.impact_model.predict_impact(x)
            return np.sum(impacts)
        except Exception:
            return float('inf')
    
    def scipy_optimize(self):
        """Solve using scipy."""
        # Initial guess
        x0 = np.full(self.n_intervals, self.total_shares / self.n_intervals)
        
        # Constraints
        constraints = [{
            'type': 'eq',
            'fun': lambda x: np.sum(x) - self.total_shares
        }]
        
        bounds = [(0, None) for _ in range(self.n_intervals)]
        
        result = minimize(
            fun=self.objective_function,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )
        
        if not result.success:
            raise Exception(f"Optimization failed: {result.message}")
        
        return result.x
    
    def cvxpy_optimize(self):
        """Solve using CVXPY."""
        x = cp.Variable(self.n_intervals, nonneg=True)
        
        # For linear model, use simple objective
        beta = self.impact_model.parameters.get('beta', 0.001)
        objective = cp.Minimize(beta * cp.sum(x))
        
        constraints = [cp.sum(x) == self.total_shares]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise Exception(f"CVXPY failed with status: {problem.status}")
        
        return x.value
    
    def optimize(self, solver=None):
        """Run optimization."""
        solver = solver or self.config.default_solver
        
        try:
            if solver == SolverType.SCIPY:
                allocation = self.scipy_optimize()
            elif solver == SolverType.CVXPY:
                allocation = self.cvxpy_optimize()
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            self.optimal_allocation = allocation
            
            # Calculate metrics
            optimal_impact = self.objective_function(allocation)
            naive_allocation = np.full(self.n_intervals, self.total_shares / self.n_intervals)
            naive_impact = self.objective_function(naive_allocation)
            improvement = ((naive_impact - optimal_impact) / naive_impact * 100) if naive_impact > 0 else 0
            
            return {
                'optimal_allocation': allocation,
                'total_impact': optimal_impact,
                'naive_impact': naive_impact,
                'improvement_pct': improvement,
                'total_shares': self.total_shares,
                'n_intervals': self.n_intervals
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
