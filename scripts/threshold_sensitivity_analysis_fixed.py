    # Validate against targets
    logger.info(f"\nValidation against targets at {best_threshold*100:.0f}%:")

    # Build target descriptions without nested quotes
    trade_freq_desc = f"Trades/Day 5-25: {best_results['trades_per_day']:.1f}"
    win_rate_desc = f"Win Rate ≥ 50%: {best_results['win_rate']:.1f}%"
    expect_desc = f"Expectation ≥ $20: ${best_results['expectation']:.2f}"
    pf_desc = f"Profit Factor ≥ 1.5: {best_results['profit_factor']:.2f}"
    sharpe_desc = f"Sharpe ≥ 0.6: {best_results['sharpe']:.2f}"
    dd_desc = f"Max Drawdown <$1K: ${best_results['max_drawdown']:.0f}"

    targets = {
        trade_freq_desc: 5 <= best_results['trades_per_day'] <= 25,
        win_rate_desc: best_results['win_rate'] >= 50.0,
        expect_desc: best_results['expectation'] >= 20.0,
        pf_desc: best_results['profit_factor'] >= 1.5,
        sharpe_desc: best_results['sharpe'] >= 0.6,
        dd_desc: best_results['max_drawdown'] < 1000.0,
    }