import logging
import os
import time
from decimal import Decimal
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import json

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent, MarketEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class AdaptivePMM(ScriptStrategyBase):
    """
    Adaptive Market Making Strategy

    This strategy extends the basic PMM with:
    1. Multi-timeframe analysis for market regime detection
    2. Dynamic spread adjustment based on volatility
    3. Trend-based price shifting
    4. Inventory management with dynamic rebalancing (target 50% ETH, 50% USDT)
    5. Risk management including position sizing, stop-loss, and circuit breakers
    """
    # Basic parameters
    trading_pair = "SOL-USDT"
    exchange = "binance_paper_trade"
    candle_exchange = 'binance'
    order_refresh_time = 20
    order_amount = 1  # Base asset amount per order
    create_timestamp = 0
    price_source = PriceType.MidPrice
    base, quote = trading_pair.split('-')

    # Multiple timeframe candles
    short_interval = "1m"
    medium_interval = "15m"
    long_interval = "1h"
    max_records = 1000

    # Market regime parameters
    regime = "neutral"  # "trending_up", "trending_down", "ranging", "volatile", "neutral", "rebounding"
    regime_change_threshold = 3  # Confirmations needed to change regime
    early_trend_detection = True

    # Moving Average parameters
    ma_enabled = True
    fast_ma_length = 15
    slow_ma_length = 45
    ma_type = "ema"

    # Spread parameters
    base_bid_spread = Decimal("0.003")  # 30 bps base spread
    base_ask_spread = Decimal("0.003")
    bid_spread = base_bid_spread
    ask_spread = base_ask_spread

    # Dynamic spread parameters (for volatility adjustment)
    volatility_adjustment_enabled = True
    # NATR multipliers for different volatility regimes
    vol_scalar_low = Decimal("1.0")  # Low volatility
    vol_scalar_medium = Decimal("1.5")  # Medium volatility
    vol_scalar_high = Decimal("2.0")  # High volatility
    current_vol_scalar = vol_scalar_medium

    max_spread_bps = 70

    # Trend-based price shift parameters
    trend_shift_enabled = True
    max_trend_shift = Decimal("0.004")  # 40 bps max shift
    trend_scalar = Decimal("0.2")  # Positive is trend-following
    price_multiplier = Decimal("1")

    # Inventory management
    inventory_management_enabled = True
    target_inventory_ratio = Decimal("0.7")  # Target 70% base, 30% quote
    uptrend_inventory_target = Decimal("0.85")  # Increase SOL exposure in uptrends
    max_uptrend_inventory = Decimal("0.95")  # Maximum SOL allocation during strong uptrends
    inventory_range_multiplier = Decimal("2")
    max_inventory_shift = Decimal("0.002")
    inventory_scalar = Decimal("1")

    # Dynamic inventory targets for different market regimes
    dynamic_inventory_targets = {
        "trending_up": Decimal("0.85"),  # Higher SOL allocation in uptrends
        "trending_down": Decimal("0.4"),  # Lower SOL allocation in downtrends
        "volatile": Decimal("0.65"),  # Balanced in volatile markets
        "ranging": Decimal("0.7"),    # Standard in ranging markets
        "rebounding": Decimal("0.75") # Start accumulating in rebounding markets as it starts to rebound
    }

    # Risk management parameters
    risk_management_enabled = True
    max_position_size = Decimal("0.1")  # Maximum position size as percentage of total portfolio
    circuit_breaker_enabled = True
    circuit_breaker_volatility_threshold = Decimal("0.05")  # 5% threshold for circuit breaker
    circuit_breaker_upside_volatility_threshold = Decimal("0.08")  # 8% for upside moves
    circuit_breaker_downside_volatility_threshold = Decimal("0.05")  # 5% for downside moves
    circuit_breaker_upside_duration = 60  # 1 minute pause for upside shocks
    circuit_breaker_downside_duration = 120  # 2 minute pause for downside shocks

    # Trailing take-profit parameters
    take_profit_enabled = True
    trailing_profit_threshold = Decimal("0.04")  # 4% price increase triggers trailing stop
    trailing_profit_distance = Decimal("0.02")  # 2% trailing distance
    trailing_profit_active = False
    trailing_reference_price = Decimal("0")
    trailing_stop_price = Decimal("0")

    # Profit ladder parameters
    profit_ladder_enabled = True
    profit_ladder_levels = 4
    profit_ladder_spacing = Decimal("0.015")  # 1.5% spacing between levels
    profit_ladder_base_multiple = Decimal("1.5")  # Base order size multiplier

    # Order tracking
    performance_tracking_enabled = True
    entry_prices = {}
    realized_pnl = Decimal("0")
    total_fees_paid = Decimal("0")
    total_trades = 0
    win_trades = 0
    loss_trades = 0

    # Order parameters
    order_levels_enabled = False
    order_levels = 1
    order_level_spread = Decimal("0.002")
    order_level_amount = Decimal("0.5")
    order_amount_buy = 1
    order_amount_sell = 1

    # Reference
    orig_price = Decimal("1")
    reference_price = Decimal("1")
    inventory_multiplier = Decimal("1")
    current_ratio = Decimal("0.5")
    inventory_delta = Decimal("0")

    # Circuit breaker
    circuit_breaker_triggered = False
    circuit_breaker_end_time = 0
    circuit_breaker_duration = 120  # 2 miutes

    # Chart visualization data
    price_history = []
    trade_history = []
    max_history_points = 500
    chart_update_interval = 15
    last_chart_update = 0
    chart_path = "sol_2_chart.html"

    markets = {exchange: {trading_pair}}

    short_candles = None
    medium_candles = None
    long_candles = None

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

        # Initialize candles for different timeframes
        self.short_candles = CandlesFactory.get_candle(CandlesConfig(connector=self.candle_exchange,
                                                      trading_pair=self.trading_pair,
                                                      interval=self.short_interval,
                                                      max_records=self.max_records))

        self.medium_candles = CandlesFactory.get_candle(CandlesConfig(connector=self.candle_exchange,
                                                       trading_pair=self.trading_pair,
                                                       interval=self.medium_interval,
                                                       max_records=self.max_records))

        self.long_candles = CandlesFactory.get_candle(CandlesConfig(connector=self.candle_exchange,
                                                     trading_pair=self.trading_pair,
                                                     interval=self.long_interval,
                                                     max_records=self.max_records))

        self.short_candles.start()
        self.medium_candles.start()
        self.long_candles.start()

        self.entry_prices = {}
        self.price_history = []
        self.trade_history = []
        self.last_chart_update = 0
        self.order_amount_buy = self.order_amount
        self.order_amount_sell = self.order_amount

        # initial_market_buy_enabled = False
        # if initial_market_buy_enabled:
        #     amount = self.order_amount
        #     try:
        #         self.buy(self.exchange, self.trading_pair, amount, OrderType.MARKET)
        #         self.log_with_clock(logging.INFO, f"Executed initial market buy of {amount} {self.base}")
        #     except Exception as e:
        #         self.log_with_clock(logging.ERROR, f"Error executing initial market buy: {str(e)}")

        self.update_chart()

        self.log_with_clock(logging.INFO, "Adaptive PMM strategy initialized!")

    def on_stop(self):
        if self.short_candles:
            self.short_candles.stop()
        if self.medium_candles:
            self.medium_candles.stop()
        if self.long_candles:
            self.long_candles.stop()

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:

            # Check if circuit breaker is active
            if self.circuit_breaker_triggered:
                if self.current_timestamp >= self.circuit_breaker_end_time:
                    self.log_with_clock(logging.INFO, "Circuit breaker deactivated, resuming trading")
                    self.circuit_breaker_triggered = False
                else:
                    self.log_with_clock(logging.INFO, f"Circuit breaker active, {self.circuit_breaker_end_time - self.current_timestamp} seconds remaining")
                    self.create_timestamp = self.order_refresh_time + self.current_timestamp
                    return

            current_price = float(self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source))
            self.price_history.append({
                'timestamp': self.current_timestamp,
                'price': current_price
            })

            if len(self.price_history) > self.max_history_points:
                self.price_history = self.price_history[-self.max_history_points:]

            if self.current_timestamp - self.last_chart_update > self.chart_update_interval:
                self.update_chart()
                self.last_chart_update = self.current_timestamp

            self.cancel_all_orders()
            self.detect_market_regime()
            self.update_strategy_parameters()

            # Manage trailing take-profit if enabled
            if self.take_profit_enabled:
                self.manage_trailing_take_profit()

            # Apply risk management
            if self.risk_management_enabled and not self.apply_risk_management():
                self.log_with_clock(logging.INFO, "Risk management prevented order placement")
                self.create_timestamp = self.order_refresh_time + self.current_timestamp
                return

            # Create order proposal
            proposal: List[OrderCandidate] = self.create_proposal()

            # Adjust to available budget
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)

            # Place orders
            self.place_orders(proposal_adjusted)

            # Setup profit-taking ladder orders in uptrends or rebounding markets
            if self.profit_ladder_enabled and self.regime in ["trending_up", "rebounding"]:
                self.setup_profit_taking_ladders()

            # Update next order refresh time
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

    def detect_market_regime(self):
        """
        Analyze multiple timeframes to determine the current market regime
        Updates self.regime with one of: trending_up, trending_down, ranging, volatile, neutral, rebounding
        """
        # Get dataframes with indicators
        short_df = self.get_candles_with_indicators(self.short_candles)
        medium_df = self.get_candles_with_indicators(self.medium_candles)
        long_df = self.get_candles_with_indicators(self.long_candles)

        if len(short_df) < 20 or len(medium_df) < 20 or len(long_df) < 20:
            self.log_with_clock(logging.INFO, "Not enough candle data for regime detection")
            return

        # Early trend reversal detection for SOL
        if self.early_trend_detection and len(short_df) > 20:
            # Check for bullish divergence (RSI making higher lows while price makes lower lows)
            price_change = short_df['close'].pct_change(3).iloc[-1]
            rsi_change = short_df['RSI_14'].diff(3).iloc[-1]

            # Volume confirmation if available
            volume_increase = False
            if 'volume' in short_df.columns:
                volume_increase = short_df['volume'].pct_change(3).iloc[-1] > 0.2  # 20% volume increase

            # Detect potential bottom formation with RSI
            oversold_recovery = (short_df['RSI_14'].iloc[-4] < 30 and short_df['RSI_14'].iloc[-1] > 40)

            # Combined signal for early trend reversal
            if ((price_change > 0.015 and rsi_change > 6) or oversold_recovery) and (volume_increase or 'volume' not in short_df.columns):
                self.log_with_clock(logging.INFO, "Detected potential trend reversal to upside")
                self.regime = "rebounding"
                return

        # Detect volatility regime
        short_vol = short_df[f"NATR_14"].iloc[-1]
        medium_vol = medium_df[f"NATR_14"].iloc[-1]
        long_vol = long_df[f"NATR_14"].iloc[-1]

        # Average volatility across timeframes
        avg_volatility = (short_vol + medium_vol + long_vol) / 3

        # Detect trend direction
        short_rsi = short_df[f"RSI_14"].iloc[-1]
        medium_rsi = medium_df[f"RSI_14"].iloc[-1]
        long_rsi = long_df[f"RSI_14"].iloc[-1]

        # ADX for trend strength
        adx_strength = medium_df[f"ADX_14"].iloc[-1] if "ADX_14" in medium_df.columns else 25

        # Moving Average trend confirmation
        ma_trend_signals = 0
        if self.ma_enabled:
            for df, weight in [(short_df, 1), (medium_df, 2), (long_df, 3)]:
                fast_col = f"{self.ma_type.upper()}_{self.fast_ma_length}"
                slow_col = f"{self.ma_type.upper()}_{self.slow_ma_length}"
                fast_slope_col = f"{fast_col}_slope"
                slow_slope_col = f"{slow_col}_slope"

                if fast_col in df.columns and slow_col in df.columns:
                    # Check if fast MA is above slow MA (bullish)
                    if df[fast_col].iloc[-1] > df[slow_col].iloc[-1]:
                        ma_trend_signals += weight
                    else:
                        ma_trend_signals -= weight

                    # Slope confirmaton
                    if fast_slope_col in df.columns and slow_slope_col in df.columns:
                        if df[fast_slope_col].iloc[-1] > 0 and df[slow_slope_col].iloc[-1] > 0:
                            ma_trend_signals += weight  # Both MAs rising (bullish)
                        elif df[fast_slope_col].iloc[-1] < 0 and df[slow_slope_col].iloc[-1] < 0:
                            ma_trend_signals -= weight  # Both MAs falling (bearish)

            self.log_with_clock(logging.DEBUG, f"MA Trend Signals: {ma_trend_signals}")

        new_regime = "neutral"

        if avg_volatility > 0.03:  # 3% threshold for high volatility
            new_regime = "volatile"
        elif adx_strength > 25:  # Strong trend detected by ADX
            # Check RSI and MAs for trend direction
            rsi_signal = 0
            if short_rsi > 55 and medium_rsi > 50 and long_rsi > 45:
                rsi_signal = 1  # Bullish RSI
            elif short_rsi < 45 and medium_rsi < 50 and long_rsi < 55:
                rsi_signal = -1  # Bearish RSI

            # Weightage: 60% RSI, 40% MA
            if self.ma_enabled and ma_trend_signals != 0:
                normalized_ma = ma_trend_signals / 6
                combined_signal = (rsi_signal * 0.6) + (normalized_ma * 0.4)

                if combined_signal > 0.3:
                    new_regime = "trending_up"
                elif combined_signal < -0.3:
                    new_regime = "trending_down"
            else:
                if rsi_signal > 0:
                    new_regime = "trending_up"
                elif rsi_signal < 0:
                    new_regime = "trending_down"
        elif 40 < medium_rsi < 60 and adx_strength < 20:
            new_regime = "ranging"

        # Check for rebounding pattern
        rebounding_pattern = False
        if len(medium_df) > 30:
            # Check if we've been in downtrend and now showing signs of reversal
            recent_low_rsi = medium_df['RSI_14'].rolling(10).min().iloc[-10:]
            recent_price = medium_df['close'].iloc[-10:]

            rsi_rising = recent_low_rsi.iloc[-1] > recent_low_rsi.iloc[0]
            price_stabilizing = recent_price.pct_change().rolling(3).sum().iloc[-1] > -0.01  # Price not falling much

            if self.regime == "trending_down" and rsi_rising and price_stabilizing:
                rebounding_pattern = True

        if rebounding_pattern:
            new_regime = "rebounding"

        if new_regime != self.regime:
            self.log_with_clock(logging.INFO, f"Market regime changed from {self.regime} to {new_regime}")
            self.regime = new_regime

    def update_strategy_parameters(self):
        """Update strategy parameters based on market regime and conditions"""
        # Update spreads based on volatility
        if self.volatility_adjustment_enabled:
            self.update_volatility_based_spreads()

        # Update price reference based on trend
        if self.trend_shift_enabled:
            self.update_trend_based_price()

        # Update inventory management
        if self.inventory_management_enabled:
            self.update_inventory_management()

        # Update order amounts based on momentum
        self.update_momentum_based_order_sizing()

    def update_momentum_based_order_sizing(self):
        """Adjust order sizes based on market momentum"""
        # Calculate momentum
        short_momentum = self.calculate_momentum("short")
        medium_momentum = self.calculate_momentum("medium")

        # Combined momentum with more weight to medium timeframe
        combined_momentum = (short_momentum * 0.4) + (medium_momentum * 0.6)

        # In uptrends, use momentum to adjust order sizes
        if self.regime in ["trending_up", "rebounding"]:
            if combined_momentum > 0:
                # Positive momentum - increase buy size, decrease sell size
                momentum_scalar = 1 + (combined_momentum * 2)  # 1.0 to 3.0 range
                self.order_amount_buy = self.order_amount * momentum_scalar
                self.order_amount_sell = self.order_amount / momentum_scalar

                self.log_with_clock(logging.INFO,
                    f"Applied momentum scalar: {momentum_scalar:.2f} (Buy: {self.order_amount_buy:.4f}, Sell: {self.order_amount_sell:.4f})")
        else:
            # Reset order amounts in other regimes
            self.order_amount_buy = self.order_amount
            self.order_amount_sell = self.order_amount

    def calculate_momentum(self, timeframe="medium"):
        """Calculate momentum score for a given timeframe"""
        if timeframe == "short":
            df = self.get_candles_with_indicators(self.short_candles)
            periods = 5
        elif timeframe == "medium":
            df = self.get_candles_with_indicators(self.medium_candles)
            periods = 3
        else:  # long
            df = self.get_candles_with_indicators(self.long_candles)
            periods = 2

        if len(df) < periods + 1:
            return 0

        # Calculate rate of change
        roc = df['close'].pct_change(periods).iloc[-1]

        # Calculate RSI momentum (change in RSI)
        rsi_momentum = df['RSI_14'].diff(periods).iloc[-1]

        # Normalized and combined momentum (-1 to 1 scale)
        normalized_roc = max(min(roc * 5, 1), -1)  # Scale ROC to -1 to 1
        normalized_rsi = rsi_momentum / 30  # Scale RSI change to approximately -1 to 1

        # Combined momentum score (70% price, 30% RSI)
        momentum = (normalized_roc * 0.7) + (normalized_rsi * 0.3)

        return momentum

    def update_volatility_based_spreads(self):
        """Adjust spread based on market volatility with asymmetric spread management"""
        # Get short timeframe candles for most recent volatility
        candles_df = self.get_candles_with_indicators(self.short_candles)

        if len(candles_df) < 14:
            return

        natr = Decimal(str(candles_df[f"NATR_14"].iloc[-1]))

        if self.regime == "volatile":
            self.current_vol_scalar = self.vol_scalar_high
        elif self.regime == "ranging":
            self.current_vol_scalar = self.vol_scalar_low
        else:
            self.current_vol_scalar = self.vol_scalar_medium

        # Update bid and ask spreads based on volatility
        base_spread = natr * self.current_vol_scalar
        self.bid_spread = base_spread
        self.ask_spread = base_spread

        # Get spread configuration
        asymmetric_spreads = getattr(self, "asymmetric_spreads", True)
        uptrend_ask_scalar = Decimal(str(getattr(self, "uptrend_ask_scalar", 0.8)))
        uptrend_bid_scalar = Decimal(str(getattr(self, "uptrend_bid_scalar", 1.2)))
        downtrend_ask_scalar = Decimal(str(getattr(self, "downtrend_ask_scalar", 1.2)))
        downtrend_bid_scalar = Decimal(str(getattr(self, "downtrend_bid_scalar", 0.8)))

        # Apply spreads based on regime
        if asymmetric_spreads:
            if self.regime == "trending_up":
                self.ask_spread = self.ask_spread * uptrend_ask_scalar
                self.bid_spread = self.bid_spread * uptrend_bid_scalar
            elif self.regime == "trending_down":
                self.bid_spread = self.bid_spread * downtrend_bid_scalar
                self.ask_spread = self.ask_spread * downtrend_ask_scalar

        if self.regime == "trending_up":
            uptrend_bid_scalar = Decimal("0.7")  # Tighter buy spreads (70% of normal)
            uptrend_ask_scalar = Decimal("1.4")  # Wider sell spreads (140% of normal)

            self.bid_spread = self.bid_spread * uptrend_bid_scalar
            self.ask_spread = self.ask_spread * uptrend_ask_scalar

            self.log_with_clock(logging.INFO, f"Applied uptrend spread adjustment: Bid scalar {uptrend_bid_scalar}, Ask scalar {uptrend_ask_scalar}")

        # For rebounding markets, use tighter buy spreads to accumulate
        if self.regime == "rebounding":
            self.bid_spread = self.bid_spread * Decimal("0.6")
            self.ask_spread = self.ask_spread * Decimal("1.3")

        # Additional adjustment based on inventory imbalance
        inventory_imbalance = abs(Decimal(str(self.current_ratio)) - self.target_inventory_ratio)
        if inventory_imbalance > Decimal("0.1"):
            if self.current_ratio > self.target_inventory_ratio:
                # Too much SOL, more aggressive on asks, less on bids
                self.ask_spread = self.ask_spread * Decimal("0.9")
                self.bid_spread = self.bid_spread * Decimal("1.2")
            else:
                # Too little SOL, more aggressive on bids, less on asks
                self.bid_spread = self.bid_spread * Decimal("0.9")
                self.ask_spread = self.ask_spread * Decimal("1.2")

        self.bid_spread = max(self.bid_spread, self.base_bid_spread)
        self.ask_spread = max(self.ask_spread, self.base_ask_spread)

        max_spread = Decimal(str(self.max_spread_bps)) / Decimal("10000")
        self.bid_spread = min(self.bid_spread, max_spread)
        self.ask_spread = min(self.ask_spread, max_spread)

    def update_trend_based_price(self):
        """Shift reference price based on trend indicators"""
        # Get medium timeframe candles for trend analysis
        candles_df = self.get_candles_with_indicators(self.medium_candles)

        if len(candles_df) < max(14, self.fast_ma_length, self.slow_ma_length):
            return

        # Get RSI for trend direction
        rsi = Decimal(str(candles_df[f"RSI_14"].iloc[-1]))

        # Price shift multiplier based on RSI deviation (50 neutral)
        rsi_shift = ((rsi - Decimal("50")) / Decimal("50")) * self.max_trend_shift * self.trend_scalar

        # MA confirmation
        ma_shift = Decimal("0")
        if self.ma_enabled:
            fast_col = f"{self.ma_type.upper()}_{self.fast_ma_length}"
            slow_col = f"{self.ma_type.upper()}_{self.slow_ma_length}"

            if fast_col in candles_df.columns and slow_col in candles_df.columns:
                fast_ma = Decimal(str(candles_df[fast_col].iloc[-1]))
                slow_ma = Decimal(str(candles_df[slow_col].iloc[-1]))

                # Calculate crossover strength
                if slow_ma != Decimal("0"):
                    ma_distance = (fast_ma - slow_ma) / slow_ma

                    ma_shift = ma_distance * self.max_trend_shift * self.trend_scalar * Decimal("0.5")
                    ma_shift = max(-self.max_trend_shift, min(self.max_trend_shift, ma_shift))

                    self.log_with_clock(logging.DEBUG,
                        f"MA Shift: {ma_shift:.6f} (Fast MA: {fast_ma:.2f}, Slow MA: {slow_ma:.2f}, Distance: {ma_distance:.4%})")

        # Combined shift - weighted average of RSI and MA signals
        # 70% weight to RSI, 30% weight to MA crossover
        self.price_multiplier = (rsi_shift * Decimal("0.7")) + (ma_shift * Decimal("0.3"))

        # If in ranging regime, reduce the price shift
        if self.regime == "ranging":
            self.price_multiplier = self.price_multiplier * Decimal("0.5")

        # If in volatile regime, more cautius with price shifts
        if self.regime == "volatile":
            self.price_multiplier = self.price_multiplier * Decimal("0.3")

        # For rebounding markets, enhance trend following
        if self.regime == "rebounding":
            self.price_multiplier = self.price_multiplier * Decimal("1.5")  # Enhanced trend following in rebounds

        self.orig_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        self.reference_price = self.orig_price * (Decimal("1") + self.price_multiplier)

    def update_inventory_management(self):
        """Adjust prices based on current inventory position with regime-based targets"""
        # Get dynamic inventory target based on market regime
        dynamic_target = self.dynamic_inventory_targets.get(self.regime, self.target_inventory_ratio)

        # Get current balances
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_bal_in_quote = base_bal * self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)

        # Calculate current inventory ratio
        total_value = base_bal_in_quote + quote_bal
        if total_value > Decimal("0"):
            self.current_ratio = Decimal(str(float(base_bal_in_quote / total_value)))
        else:
            self.current_ratio = dynamic_target  # Default to dynamic target if no inventory

        # Calculate deviation from target
        delta = ((dynamic_target - self.current_ratio) / dynamic_target)
        self.inventory_delta = max(Decimal("-1"), min(Decimal("1"), delta))

        self.log_with_clock(logging.INFO,
                            f"Inventory: Target={float(dynamic_target):.2f} Current={float(self.current_ratio):.2f} Delta={float(self.inventory_delta):.2f}")

        # Calculate inventory price shift
        self.inventory_multiplier = self.inventory_delta * self.max_inventory_shift * self.inventory_scalar

        # Apply stronger inventory management in ranging markets, gentler in trending
        if self.regime == "ranging":
            self.inventory_multiplier = self.inventory_multiplier * Decimal("1.2")  # More aggressive in ranging
        elif self.regime == "trending_up" or self.regime == "trending_down":
            self.inventory_multiplier = self.inventory_multiplier * Decimal("0.7")  # Less aggressive in trending

        # Apply inventory shift to reference price
        self.reference_price = self.reference_price * (Decimal("1") + self.inventory_multiplier)

    def manage_trailing_take_profit(self):
        """Implement trailing take-profit mechanism for uptrends"""
        if not self.take_profit_enabled or self.regime != "trending_up" or self.regime !="rebounding":
            return

        current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)

        if not self.trailing_profit_active:
            # Check if price has increased enough to activate trailing take-profit
            if self.trailing_reference_price == Decimal("0"):
                self.trailing_reference_price = current_price
                return

            price_increase = (current_price - self.trailing_reference_price) / self.trailing_reference_price
            if price_increase > self.trailing_profit_threshold:
                self.trailing_profit_active = True
                self.trailing_stop_price = current_price * (Decimal("1") - self.trailing_profit_distance)
                self.log_with_clock(logging.INFO, f"Activated trailing take-profit at {float(current_price)}")
        else:
            # Manage active trailing take-profit
            trailing_sell_price = current_price * (Decimal("1") - self.trailing_profit_distance)

            # Place a larger market sell if price drops below trailing stop level
            if current_price < self.trailing_stop_price and self.trailing_stop_price > Decimal("0"):
                base_balance = self.connectors[self.exchange].get_balance(self.base)
                sell_amount = min(base_balance * Decimal("0.3"), Decimal(str(self.order_amount * 3)))

                if sell_amount > Decimal("0"):
                    self.sell(self.exchange, self.trading_pair, sell_amount, OrderType.MARKET)
                    self.log_with_clock(logging.INFO,
                        f"Trailing take-profit triggered: Sold {float(sell_amount)} {self.base} at {float(current_price)}")
                    self.trailing_profit_active = False
                    self.trailing_reference_price = Decimal("0")
            else:
                # Update trailing stop price if price continues to rise
                self.trailing_stop_price = max(trailing_sell_price, self.trailing_stop_price)

    def setup_profit_taking_ladders(self):
        """Set up ladder of sell orders at progressively higher prices during uptrends"""
        if not self.profit_ladder_enabled or self.regime not in ["trending_up", "rebounding"]:
            return

        current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        base_balance = self.connectors[self.exchange].get_balance(self.base)

        # Calculate available amount for profit ladders (30% of base balance)
        available_amount = base_balance * Decimal("0.3")

        # Skip if not enough balance
        if available_amount < Decimal(str(self.order_amount)):
            return

        # Create ladder of sell orders at progressively higher prices
        level_amount = available_amount / Decimal(str(self.profit_ladder_levels))

        for i in range(1, self.profit_ladder_levels + 1):
            # Calculate price level with increasing spacing
            level_spacing = self.profit_ladder_spacing * Decimal(str(i))
            price_level = current_price * (Decimal("1") + level_spacing)

            # Calculate size for this level (higher prices get larger sizes)
            size_multiple = self.profit_ladder_base_multiple * (Decimal("1") + Decimal("0.2") * Decimal(str(i-1)))
            level_size = min(level_amount * size_multiple, available_amount)

            # Place the sell order
            self.sell(self.exchange, self.trading_pair, level_size, OrderType.LIMIT, price_level)
            self.log_with_clock(logging.INFO,
                f"Placed profit ladder sell order: {float(level_size)} {self.base} at {float(price_level)}")

    def apply_risk_management(self) -> bool:
        """
        Apply risk management rules
        Returns True if orders should be placed, False if orders should be skipped
        """
        if not self.risk_management_enabled:
            return True

        if self.circuit_breaker_enabled:
            # Get recent volatility
            short_df = self.get_candles_with_indicators(self.short_candles)
            if len(short_df) >= 3:
                # Calculate recent price change
                recent_change = short_df['close'].pct_change(2).iloc[-1]

                # Different thresholds for upside vs downside volatility
                if recent_change > 0:  # Price increase
                    threshold = self.circuit_breaker_upside_volatility_threshold
                    duration = self.circuit_breaker_upside_duration
                else:  # Price decrease
                    threshold = self.circuit_breaker_downside_volatility_threshold
                    duration = self.circuit_breaker_downside_duration

                if abs(recent_change) > float(threshold):
                    self.log_with_clock(logging.WARNING,
                        f"Circuit breaker triggered! Recent price change of {recent_change:.2%}")
                    self.circuit_breaker_triggered = True
                    self.circuit_breaker_end_time = self.current_timestamp + duration
                    return False

        # Get portfolio value
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        base_value = base_bal * base_price
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        total_portfolio_value = base_value + quote_bal

        # Check if proposed order amount exceeds max position size
        order_value = Decimal(str(self.order_amount)) * base_price
        max_allowed_value = total_portfolio_value * self.max_position_size

        if order_value > max_allowed_value:
            # Reduce order size
            adjusted_amount = max_allowed_value / base_price
            self.log_with_clock(logging.INFO,
                f"Order amount reduced from {self.order_amount} to {float(adjusted_amount)} due to position size limits")
            self.order_amount = float(adjusted_amount)
            self.order_amount_buy = float(adjusted_amount)
            self.order_amount_sell = float(adjusted_amount)

        return True

    def get_candles_with_indicators(self, candles):
        candles_df = candles.candles_df.copy()

        if len(candles_df) < max(14, self.fast_ma_length, self.slow_ma_length):
            return candles_df

        # Normalized Average True Range for volatility
        candles_df.ta.natr(length=14, scalar=1, append=True)

        # RSI for trend direction
        candles_df.ta.rsi(length=14, append=True)

        # ADX for trend strength
        candles_df.ta.adx(length=14, append=True)

        # MACD for trend confirmation
        candles_df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # Bollinger Bands for range detection
        candles_df.ta.bbands(length=20, std=2, append=True)

        # Moving Averages for trend confirmation
        if self.ma_enabled:
            if self.ma_type == "sma":
                # Simple Moving Average
                candles_df.ta.sma(length=self.fast_ma_length, append=True)
                candles_df.ta.sma(length=self.slow_ma_length, append=True)
            elif self.ma_type == "wma":
                # Weighted Moving Average
                candles_df.ta.wma(length=self.fast_ma_length, append=True)
                candles_df.ta.wma(length=self.slow_ma_length, append=True)
            else:
                # Exponential Moving Average (default)
                candles_df.ta.ema(length=self.fast_ma_length, append=True)
                candles_df.ta.ema(length=self.slow_ma_length, append=True)

            # Calculate MA crossover signal
            fast_col = f"{self.ma_type.upper()}_{self.fast_ma_length}"
            slow_col = f"{self.ma_type.upper()}_{self.slow_ma_length}"

            # MA Crossover (1 for bullish, -1 for bearish, 0 for no cross)
            candles_df['ma_cross'] = 0

            if len(candles_df) >= 2:
                # Current crossover state
                candles_df.loc[candles_df.index[-1], 'ma_cross'] = 1 if candles_df[fast_col].iloc[-1] > candles_df[slow_col].iloc[-1] else -1

                # Previous state
                candles_df.loc[candles_df.index[-2], 'ma_prev_cross'] = 1 if candles_df[fast_col].iloc[-2] > candles_df[slow_col].iloc[-2] else -1

                # Detect crossover
                if 'ma_prev_cross' in candles_df.columns:
                    candles_df['ma_crossover_signal'] = (candles_df['ma_cross'] != candles_df['ma_prev_cross']).astype(int) * candles_df['ma_cross']

            # Calculate MA slopes for trend strength (last 5 periods)
            if len(candles_df) >= 5:
                candles_df[f'{fast_col}_slope'] = (candles_df[fast_col] - candles_df[fast_col].shift(5)) / 5
                candles_df[f'{slow_col}_slope'] = (candles_df[slow_col] - candles_df[slow_col].shift(5)) / 5

        return candles_df

    def create_proposal(self) -> List[OrderCandidate]:
        if self.order_levels_enabled and self.order_levels > 1:
            return self.create_multi_level_orders()
        else:
            return self.create_single_level_orders()

    def create_single_level_orders(self) -> List[OrderCandidate]:
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)

        buy_price = min(self.reference_price * (Decimal("1") - self.bid_spread), best_bid)
        sell_price = max(self.reference_price * (Decimal("1") + self.ask_spread), best_ask)

        # Position Scaling During Strong Uptrends
        if self.regime == "trending_up":
            # Check momentum strength
            medium_df = self.get_candles_with_indicators(self.medium_candles)
            if len(medium_df) >= 3:
                momentum = medium_df['close'].pct_change(3).iloc[-1]

                if momentum > 0.05:  # Strong momentum (5%+ in medium timeframe)
                    # Calculate buy order size multiplier based on momentum strength
                    buy_size_multiplier = min(Decimal("3.0"), Decimal(str(1.0 + momentum * 10)))

                    # Increase buy order size, decrease sell order size
                    self.order_amount_buy = float(Decimal(str(self.order_amount)) * buy_size_multiplier)
                    self.order_amount_sell = float(Decimal(str(self.order_amount)) * Decimal("0.5"))

                    self.log_with_clock(logging.INFO,
                        f"Applying position scaling in strong uptrend: Buy multiplier {float(buy_size_multiplier)}")

        buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                order_side=TradeType.BUY, amount=Decimal(str(self.order_amount_buy)), price=buy_price)

        sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=Decimal(str(self.order_amount_sell)), price=sell_price)

        dynamic_target = self.dynamic_inventory_targets.get(self.regime, self.target_inventory_ratio)

        current_base_bal = self.connectors[self.exchange].get_balance(self.base)
        current_quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)

        # Adjust order amounts based on inventory imbalance
        if self.inventory_management_enabled and abs(self.inventory_delta) > Decimal("0.2"):
            if self.inventory_delta > Decimal("0"):
                # Need more base asset - place only buy order if severe imbalance
                if self.inventory_delta > Decimal("0.8"):
                    self.log_with_clock(logging.INFO, f"Significant inventory imbalance: {float(self.inventory_delta):.2f}, only placing buy orders")
                    return [buy_order]

                buy_amount = Decimal(str(self.order_amount_buy)) * (Decimal("1") + min(self.inventory_delta, Decimal("0.5")))
                sell_amount = Decimal(str(self.order_amount_sell)) * (Decimal("1") - min(self.inventory_delta, Decimal("0.5")))
                buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                        order_side=TradeType.BUY, amount=buy_amount, price=buy_price)
                sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                            order_side=TradeType.SELL, amount=sell_amount, price=sell_price)
            else:
                # Need less base asset - place only sell order if severe imbalance
                if self.inventory_delta < Decimal("-0.8"):
                    self.log_with_clock(logging.INFO, f"Significant inventory imbalance: {float(self.inventory_delta):.2f}, only placing sell orders")
                    return [sell_order]

                sell_amount = Decimal(str(self.order_amount_sell)) * (Decimal("1") + min(abs(self.inventory_delta), Decimal("0.5")))
                buy_amount = Decimal(str(self.order_amount_buy)) * (Decimal("1") - min(abs(self.inventory_delta), Decimal("0.5")))
                buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                        order_side=TradeType.BUY, amount=buy_amount, price=buy_price)
                sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                            order_side=TradeType.SELL, amount=sell_amount, price=sell_price)

        return [buy_order, sell_order]

    def create_multi_level_orders(self) -> List[OrderCandidate]:
        order_candidates = []

        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)

        for level in range(self.order_levels):
            level_bid_spread = self.bid_spread * (Decimal("1") + level * self.order_level_spread)
            level_ask_spread = self.ask_spread * (Decimal("1") + level * self.order_level_spread)
            level_amount = Decimal(str(self.order_amount)) * (Decimal("1") + level * self.order_level_amount)

            buy_price = min(self.reference_price * (Decimal("1") - level_bid_spread), best_bid)
            sell_price = max(self.reference_price * (Decimal("1") + level_ask_spread), best_ask)

            buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                      order_side=TradeType.BUY, amount=level_amount, price=buy_price)

            sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                       order_side=TradeType.SELL, amount=level_amount, price=sell_price)

            if self.inventory_management_enabled and abs(self.inventory_delta) > Decimal("0.8"):
                if self.inventory_delta > Decimal("0"):
                    order_candidates.append(buy_order)
                else:
                    order_candidates.append(sell_order)
            else:
                order_candidates.extend([buy_order, sell_order])

        return order_candidates

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)
        return proposal_adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        order_id = event.order_id
        filled_amount = event.amount
        executed_price = event.price
        trade_type = event.trade_type.name
        fee = event.trade_fee.get_fee_impact_on_order_cost(True, self.exchange)

        self.total_trades += 1
        self.total_fees_paid += fee

        self.trade_history.append({
            'timestamp': self.current_timestamp,
            'price': float(executed_price),
            'amount': float(filled_amount),
            'type': trade_type
        })

        if len(self.trade_history) > self.max_history_points:
            self.trade_history = self.trade_history[-self.max_history_points:]

        if order_id not in self.entry_prices:
            self.entry_prices[order_id] = executed_price

        # DO NOT FORGET - Implement PNL metrrics for end
        if self.performance_tracking_enabled:
            pass

        msg = (f"{trade_type} {round(float(filled_amount), 4)} {event.trading_pair} at {round(float(executed_price), 4)}, "
               f"Fee: {round(float(fee), 6)} {self.quote}, Regime: {self.regime}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        self.update_chart()

    def update_chart(self):
        try:
            timestamps = [p['timestamp'] for p in self.price_history]
            prices = [p['price'] for p in self.price_history]
            buy_trades = [{'x': t['timestamp'], 'y': t['price']} for t in self.trade_history if t['type'] == 'BUY']
            sell_trades = [{'x': t['timestamp'], 'y': t['price']} for t in self.trade_history if t['type'] == 'SELL']

            current_orders = self.get_active_orders(connector_name=self.exchange)
            buy_orders = [{'price': float(o.price), 'amount': float(o.quantity)}
                         for o in current_orders if o.is_buy]
            sell_orders = [{'price': float(o.price), 'amount': float(o.quantity)}
                          for o in current_orders if not o.is_buy]

            self.log_with_clock(logging.DEBUG, f"Active buy orders: {len(buy_orders)}, Active sell orders: {len(sell_orders)}")
            for i, order in enumerate(buy_orders):
                self.log_with_clock(logging.DEBUG, f"Buy order {i}: Price={order['price']}")
            for i, order in enumerate(sell_orders):
                self.log_with_clock(logging.DEBUG, f"Sell order {i}: Price={order['price']}")

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Hummingbot Trading Chart</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
                <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
                <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@1.2.1"></script>
                <meta http-equiv="refresh" content="15">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    #chart-container {{ width: 100%; height: 600px; }}
                    .stats {{ margin-top: 20px; }}
                    .timestamp {{ color: #666; font-size: 12px; text-align: right; }}
                </style>
            </head>
            <body>
                <h1>Trading Chart: {self.trading_pair}</h1>
                <div id="chart-container">
                    <canvas id="tradingChart"></canvas>
                </div>
                <div class="stats">
                    <h3>Strategy Status</h3>
                    <p>Market Regime: <b>{self.regime}</b></p>
                    <p>Current Ratio: <b>{float(self.current_ratio):.2%}</b> (Target: {float(self.target_inventory_ratio):.2%})</p>
                    <p>Spreads: Bid {float(self.bid_spread)*10000:.2f} bps | Ask {float(self.ask_spread)*10000:.2f} bps</p>

                    <h3>Open Orders</h3>
                    <table border="1" style="border-collapse: collapse; width: 100%">
                        <tr>
                            <th>Side</th>
                            <th>Amount</th>
                            <th>Price</th>
                            <th>Spread</th>
                        </tr>
                        {"".join([f'<tr style="background-color: rgba(75, 192, 192, 0.2)"><td>BUY</td><td>{o["amount"]:.6f}</td><td>{o["price"]:.2f}</td><td>{((float(self.orig_price) - o["price"])/float(self.orig_price)*10000):.2f} bps</td></tr>' for o in buy_orders])}
                        {"".join([f'<tr style="background-color: rgba(255, 99, 132, 0.2)"><td>SELL</td><td>{o["amount"]:.6f}</td><td>{o["price"]:.2f}</td><td>{((o["price"] - float(self.orig_price))/float(self.orig_price)*10000):.2f} bps</td></tr>' for o in sell_orders])}
                    </table>
                    <p class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <script>
                    const ctx = document.getElementById('tradingChart').getContext('2d');

                    // Price history data
                    const priceData = {json.dumps(list(zip(timestamps, prices)))};

                    // Format for Chart.js
                    const prices_formatted = priceData.map(p => ({{
                        x: p[0] * 1000, // Convert to milliseconds
                        y: p[1]
                    }}));

                    // Trade data
                    const buyTrades = {json.dumps(buy_trades)};
                    const sellTrades = {json.dumps(sell_trades)};

                    // Format trade data for Chart.js
                    const buyTrades_formatted = buyTrades.map(t => ({{
                        x: t.x * 1000, // Convert to milliseconds
                        y: t.y
                    }}));

                    const sellTrades_formatted = sellTrades.map(t => ({{
                        x: t.x * 1000, // Convert to milliseconds
                        y: t.y
                    }}));

                    // Current orders
                    const buyOrders = {json.dumps(buy_orders)};
                    const sellOrders = {json.dumps(sell_orders)};

                    // Create datasets for all charts
                    const datasets = [
                        {{
                            label: 'Price',
                            data: prices_formatted,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            pointRadius: 0,
                            borderWidth: 1,
                            fill: false
                        }},
                        {{
                            label: 'Buy Trades',
                            data: buyTrades_formatted,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 1)',
                            pointRadius: 6,
                            pointStyle: 'triangle',
                            pointRotation: 180,
                            showLine: false
                        }},
                        {{
                            label: 'Sell Trades',
                            data: sellTrades_formatted,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 1)',
                            pointRadius: 6,
                            pointStyle: 'triangle',
                            showLine: false
                        }}
                    ];

                    // Get min and max timestamps for horizontal lines
                    let minTime = null;
                    let maxTime = null;

                    if (prices_formatted.length > 0) {{
                        const allTimes = prices_formatted.map(p => p.x);
                        minTime = Math.min(...allTimes);
                        maxTime = Math.max(...allTimes);
                    }} else {{
                        // Default range if no price data
                        const now = new Date().getTime();
                        minTime = now - 1800000; // 30 min ago
                        maxTime = now;
                    }}

                    // Add horizontal lines for buy orders
                    if (buyOrders && buyOrders.length > 0) {{
                        buyOrders.forEach((order, index) => {{
                            datasets.push({{
                                label: `Buy @ ${{order.price.toFixed(2)}}`,
                                data: [
                                    {{ x: minTime, y: order.price }},
                                    {{ x: maxTime, y: order.price }}
                                ],
                                borderColor: 'rgba(75, 192, 192, 0.8)',
                                backgroundColor: 'transparent',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                pointRadius: 0,
                                fill: false,
                                order: 1 // Lower order means it's drawn first (under other elements)
                            }});
                        }});
                    }}

                    // Add horizontal lines for sell orders
                    if (sellOrders && sellOrders.length > 0) {{
                        sellOrders.forEach((order, index) => {{
                            datasets.push({{
                                label: `Sell @ ${{order.price.toFixed(2)}}`,
                                data: [
                                    {{ x: minTime, y: order.price }},
                                    {{ x: maxTime, y: order.price }}
                                ],
                                borderColor: 'rgba(255, 99, 132, 0.8)',
                                backgroundColor: 'transparent',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                pointRadius: 0,
                                fill: false,
                                order: 1
                            }});
                        }});
                    }}

                    // Create the chart
                    const tradingChart = new Chart(ctx, {{
                        type: 'line',
                        data: {{ datasets: datasets }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: {{
                                mode: 'index',
                                intersect: false,
                            }},
                            scales: {{
                                x: {{
                                    type: 'time',
                                    time: {{
                                        unit: 'minute',
                                        tooltipFormat: 'MMM d, HH:mm:ss'
                                    }},
                                    title: {{
                                        display: true,
                                        text: 'Time'
                                    }}
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'Price ({self.quote})'
                                    }}
                                }}
                            }},
                            plugins: {{
                                zoom: {{
                                    pan: {{
                                        enabled: true,
                                        mode: 'x'
                                    }},
                                    zoom: {{
                                        wheel: {{
                                            enabled: true,
                                        }},
                                        pinch: {{
                                            enabled: true
                                        }},
                                        mode: 'x',
                                    }}
                                }},
                                legend: {{
                                    display: true,
                                    position: 'top'
                                }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """

            with open(self.chart_path, 'w') as f:
                f.write(html_content)

            self.log_with_clock(logging.INFO, f"Chart updated at {self.chart_path}")
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"Error updating chart: {str(e)}")

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([f"  Strategy State"])
        lines.extend([f"  Market Regime: {self.regime}"])
        lines.extend([f"  Chart available at: {os.path.abspath(self.chart_path)}"])

        # if self.performance_tracking_enabled:
            # lines.extend([f"  Total Trades: {self.total_trades}"])
            # lines.extend([f"  Total Fees Paid: {round(self.total_fees_paid, 6)} {self.quote}"])
            # lines.extend([f"  Realized PnL: {round(self.realized_pnl, 6)} {self.quote}"])

        if self.risk_management_enabled:
            circuit_breaker_status = "ACTIVE" if self.circuit_breaker_triggered else "Inactive"
            lines.extend([f"  Circuit Breaker: {circuit_breaker_status}"])

        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend(["  Spreads:"])
        lines.extend([f"  Bid Spread: {float(self.bid_spread) * 10000:.2f} bps | Ask Spread: {float(self.ask_spread) * 10000:.2f} bps"])
        lines.extend([f"  Current Vol Scalar: {float(self.current_vol_scalar)}"])

        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend(["  Price Shifts:"])

        trend_price_shift = self.price_multiplier * self.reference_price
        inventory_price_shift = self.inventory_multiplier * self.reference_price

        lines.extend([f"  Trend Multiplier (bps): {float(self.price_multiplier) * 10000:.2f} | Trend Price Shift: {float(trend_price_shift):.4f}"])
        lines.extend([f"  Target Inventory Ratio: {float(self.target_inventory_ratio):.2f} | Current Ratio: {float(self.current_ratio):.2f}"])
        lines.extend([f"  Inventory Multiplier (bps): {float(self.inventory_multiplier) * 10000:.2f} | Inventory Price Shift: {float(inventory_price_shift):.4f}"])
        lines.extend([f"  Orig Price: {float(self.orig_price):.4f} | Reference Price: {float(self.reference_price):.4f}"])

        try:
            lines.extend(["\n----------------------------------------------------------------------\n"])
            lines.extend([f"  Short Candles ({self.short_interval}):"])
            short_df = self.get_candles_with_indicators(self.short_candles)
            if len(short_df) > 0:
                latest = short_df.iloc[-1]
                lines.extend([f"  RSI: {latest.get('RSI_14', 'N/A'):.1f} | NATR: {latest.get('NATR_14', 'N/A'):.6f} | ADX: {latest.get('ADX_14', 'N/A'):.1f}"])

                if self.ma_enabled:
                    fast_col = f"{self.ma_type.upper()}_{self.fast_ma_length}"
                    slow_col = f"{self.ma_type.upper()}_{self.slow_ma_length}"
                    if fast_col in latest and slow_col in latest:
                        fast_ma = latest.get(fast_col, 'N/A')
                        slow_ma = latest.get(slow_col, 'N/A')
                        ma_state = "BULLISH" if fast_ma > slow_ma else "BEARISH"
                        lines.extend([f"  {fast_col}: {fast_ma:.2f} | {slow_col}: {slow_ma:.2f} | State: {ma_state}"])

                        fast_slope_col = f"{fast_col}_slope"
                        slow_slope_col = f"{slow_col}_slope"
                        if fast_slope_col in latest and slow_slope_col in latest:
                            fast_slope = latest.get(fast_slope_col, 'N/A')
                            slow_slope = latest.get(slow_slope_col, 'N/A')
                            fast_dir = "UP" if fast_slope > 0 else "DOWN"
                            slow_dir = "UP" if slow_slope > 0 else "DOWN"
                            lines.extend([f"  {fast_col} Slope: {fast_slope:.6f} ({fast_dir}) | {slow_col} Slope: {slow_slope:.6f} ({slow_dir})"])

            lines.extend([f"  Medium Candles ({self.medium_interval}):"])
            medium_df = self.get_candles_with_indicators(self.medium_candles)
            if len(medium_df) > 0:
                latest = medium_df.iloc[-1]
                lines.extend([f"  RSI: {latest.get('RSI_14', 'N/A'):.1f} | NATR: {latest.get('NATR_14', 'N/A'):.6f} | ADX: {latest.get('ADX_14', 'N/A'):.1f}"])

                if self.ma_enabled:
                    fast_col = f"{self.ma_type.upper()}_{self.fast_ma_length}"
                    slow_col = f"{self.ma_type.upper()}_{self.slow_ma_length}"
                    if fast_col in latest and slow_col in latest:
                        fast_ma = latest.get(fast_col, 'N/A')
                        slow_ma = latest.get(slow_col, 'N/A')
                        ma_state = "BULLISH" if fast_ma > slow_ma else "BEARISH"
                        lines.extend([f"  {fast_col}: {fast_ma:.2f} | {slow_col}: {slow_ma:.2f} | State: {ma_state}"])
        except Exception as e:
            lines.extend([f"  Error loading indicator data: {str(e)}"])

        return "\n".join(lines)
