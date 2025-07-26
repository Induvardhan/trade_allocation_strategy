"""
Simplified production-ready impact model for demonstration.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for production model."""
    base_price: float = 100.0
    memory_limit_mb: int = 1000
    progress_logging: bool = True

class ValidationError(Exception):
    """Validation error exception."""
    pass

class ImpactModel:
    """Production-ready impact model."""
    
    def __init__(self, model_type='linear', config: Optional[ModelConfig] = None):
        self.model_type = model_type
        self.config = config or ModelConfig()
        self.fitted_model = None
        self.parameters = {}
        logger.info(f"Initialized ImpactModel with type: {self.model_type}")
    
    def generate_synthetic_orderbook(self, n_snapshots=100, n_levels=10):
        """Generate synthetic orderbook data."""
        np.random.seed(42)
        data = []
        
        for t in range(n_snapshots):
            mid_price = self.config.base_price + 0.1 * t
            for side in ['bid', 'ask']:
                for level in range(n_levels):
                    if side == 'bid':
                        price = mid_price - 0.01 * (level + 1)
                    else:
                        price = mid_price + 0.01 * (level + 1)
                    
                    size = 1000 * np.exp(-0.1 * level) * (1 + 0.2 * np.random.normal())
                    
                    data.append({
                        'time': t,
                        'side': side,
                        'price': price,
                        'size': max(100, size),
                        'level': level
                    })
        
        return pd.DataFrame(data)
    
    def estimate_slippage(self, orderbook_data):
        """Estimate slippage from orderbook data."""
        trade_sizes = np.logspace(2, 4, 20)  # 100 to 10,000 shares
        slippage_data = []
        
        for t in orderbook_data['time'].unique():
            snapshot = orderbook_data[orderbook_data['time'] == t]
            asks = snapshot[snapshot['side'] == 'ask'].sort_values('price')
            bids = snapshot[snapshot['side'] == 'bid'].sort_values('price', ascending=False)
            
            if len(asks) == 0 or len(bids) == 0:
                continue
                
            mid_price = (bids.iloc[0]['price'] + asks.iloc[0]['price']) / 2
            
            for size in trade_sizes:
                # Simple slippage calculation
                buy_slippage = size * 0.001  # 0.1% impact per 1000 shares
                sell_slippage = size * 0.001
                avg_slippage = (buy_slippage + sell_slippage) / 2
                
                slippage_data.append({
                    'time': t,
                    'trade_size': size,
                    'avg_slippage': avg_slippage,
                    'mid_price': mid_price
                })
        
        return pd.DataFrame(slippage_data)
    
    def fit_model(self, slippage_data):
        """Fit linear impact model."""
        X = slippage_data['trade_size'].values.reshape(-1, 1)
        y = slippage_data['avg_slippage'].values
        
        self.fitted_model = LinearRegression()
        self.fitted_model.fit(X, y)
        
        predictions = self.fitted_model.predict(X)
        r2 = r2_score(y, predictions)
        
        self.parameters = {
            'beta': self.fitted_model.coef_[0],
            'intercept': self.fitted_model.intercept_
        }
        
        logger.info(f"Model fitted with R² = {r2:.4f}")
        
        return {
            'model': self.fitted_model,
            'metrics': {'test_r2': r2, 'train_r2': r2},
            'data_stats': {'total_points': len(slippage_data)}
        }
    
    def predict_impact(self, trade_sizes):
        """Predict impact for given trade sizes."""
        if self.fitted_model is None:
            raise ValidationError("Model not fitted. Call fit_model() first.")
        
        if np.isscalar(trade_sizes):
            trade_sizes = np.array([trade_sizes])
        else:
            trade_sizes = np.array(trade_sizes)
        
        return self.fitted_model.predict(trade_sizes.reshape(-1, 1))

if __name__ == "__main__":
    # Simple test
    model = ImpactModel()
    data = model.generate_synthetic_orderbook(50, 5)
    slippage = model.estimate_slippage(data)
    results = model.fit_model(slippage)
    print(f"Test completed. R² = {results['metrics']['test_r2']:.4f}")
