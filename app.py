"""AviatorLab + MatchWise: educational Streamlit dashboard."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from aviatorlab_matchwise.aviator import (  # noqa: E402
    estimate_cashout_rtp,
    fair_game_comparison,
    independence_test,
    sample_crash_points,
    simulate_historical_cashout,
    simulate_random_bankroll,
    summarize_multipliers,
)
from aviatorlab_matchwise.loaders import load_fixtures, load_matches, load_multipliers  # noqa: E402
from aviatorlab_matchwise.matchwise import PoissonMatchModel, walk_forward_evaluation  # noqa: E402

st.set_page_config(page_title="AviatorLab + MatchWise", page_icon="📊", layout="wide")


@st.cache_data
def sample_multipliers():
    return load_multipliers(ROOT / "data" / "sample_aviator_multipliers.csv")


@st.cache_data
def sample_matches():
    return load_matches(ROOT / "data" / "sample_matches.csv")


@st.cache_data
def cached_backtest(frame: pd.DataFrame, shrinkage: float):
    return walk_forward_evaluation(frame, min_train=20, step=1, shrinkage=shrinkage)


with st.sidebar:
    st.header("Your profile")
    name = st.text_input("Display name", value="Guest", max_chars=20).strip() or "Guest"
    loss_limit = st.number_input("Session loss limit (credits)", 10.0, 100000.0, 200.0, 10.0)
    st.caption("The live simulator stops by itself when this limit is reached.")
    st.divider()
    st.caption("Virtual credits only. No real money, accounts or betting data.")

st.title(f"AviatorLab + MatchWise · {name}")
st.caption("Open-source statistical analysis, sports probabilities, and responsible-play education.")
st.warning("Educational prototype only. Aviator history cannot reliably predict the next crash point. "
           "Football probabilities are uncertain estimates, not guaranteed outcomes or betting advice.")
live_tab, aviator_tab, fixtures_tab, match_tab, about_tab = st.tabs(
    ["Live lab", "AviatorLab", "Fixtures", "MatchWise", "Responsible play"]
)

# --------------------------------------------------------------------------- #
with live_tab:
    st.subheader("Live crash simulator (virtual credits)")
    st.caption("A new random round every 2 seconds from a 97%-RTP model. This is a simulation, not a feed "
               "from any real game, and past rounds never predict the next one.")
    ss = st.session_state
    ss.setdefault("live_on", False)
    ss.setdefault("wallet", 1000.0)
    ss.setdefault("rounds", [])
    ss.setdefault("path", [1000.0])
    ss.setdefault("rng", np.random.default_rng())
    ss.setdefault("start_wallet", 1000.0)

    l1, l2, l3, l4 = st.columns([1, 1, 1, 1])
    live_stake = l1.number_input("Stake", 1.0, 1000.0, 10.0, 1.0, key="live_stake")
    live_target = l2.number_input("Auto cash-out (×)", 1.1, 100.0, 2.0, 0.1, key="live_target")
    l3.write("")
    if l3.button("Stop" if ss.live_on else "Start live rounds", type="primary"):
        ss.live_on = not ss.live_on
        if ss.live_on:
            ss.start_wallet = ss.wallet
        st.rerun()
    l4.write("")
    if l4.button("Reset wallet"):
        ss.update(live_on=False, wallet=1000.0, rounds=[], path=[1000.0], start_wallet=1000.0)
        st.rerun()

    def live_feed():
        if ss.live_on:
            crash = float(sample_crash_points(1, rng=ss.rng)[0])
            if ss.start_wallet - ss.wallet >= loss_limit or ss.wallet < live_stake:
                ss.live_on = False
                st.rerun()  # full rerun so the button label and message update
            ss.wallet += live_stake * (live_target - 1) if crash >= live_target else -live_stake
            ss.rounds = (ss.rounds + [crash])[-300:]
            ss.path = (ss.path + [ss.wallet])[-300:]
        rounds = ss.rounds
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Virtual balance", f"{ss.wallet:,.0f}", f"{ss.wallet - 1000:+,.0f} vs 1,000")
        m2.metric("Rounds", len(rounds))
        hit2 = (np.array(rounds) >= 2).mean() if rounds else None
        m3.metric("Reached 2× or more", f"{hit2:.1%}" if hit2 is not None else "–", "model 48.5%", delta_color="off")
        m4.metric("Last crash", f"{rounds[-1]:.2f}×" if rounds else "–")
        if rounds:
            st.markdown(" ".join(f"`{c:.2f}×`" for c in reversed(rounds[-14:])))
            st.line_chart(pd.DataFrame({"Balance": ss.path}), height=220)
        else:
            st.info("Press **Start live rounds** to begin. Your balance will drift toward the house edge over time.")
        if not ss.live_on and ss.start_wallet - ss.wallet >= loss_limit:
            st.warning("Loss limit reached, so the simulator stopped. That is the limit doing its job.")

    st.fragment(run_every=2 if ss.live_on else None)(live_feed)()

# --------------------------------------------------------------------------- #
with fixtures_tab:
    st.subheader("Upcoming fixtures")
    st.caption("Demo fixtures between fictional teams. Click a match for the model's probabilities. "
               "Upload your own fixtures CSV (home_team, away_team, date or day_offset, optional kickoff).")
    fx_up = st.file_uploader("Optional fixtures CSV", type=["csv"], key="fx")
    try:
        fixtures = load_fixtures(fx_up) if fx_up else load_fixtures(ROOT / "data" / "sample_fixtures.csv")
    except ValueError as exc:
        st.error(str(exc))
        fixtures = None
    fx_model = PoissonMatchModel().fit(sample_matches())

    @st.dialog("Match preview")
    def preview(home: str, away: str):
        pred = fx_model.predict_match(home, away)
        st.markdown(f"### {home} v {away}")
        a, b, c = st.columns(3)
        a.metric(home, f"{pred['home_win']:.0%}")
        b.metric("Draw", f"{pred['draw']:.0%}")
        c.metric(away, f"{pred['away_win']:.0%}")
        if pred["unseen_teams"]:
            st.warning(f"No history for {', '.join(pred['unseen_teams'])}; treated as an average team.")
        st.caption(f"Expected goals {pred['home_goal_mean']:.2f} to {pred['away_goal_mean']:.2f}")
        grid = pd.DataFrame(pred["score_matrix"][:6, :6]).round(3)
        grid.index.name, grid.columns.name = "Home goals", "Away goals"
        st.plotly_chart(px.imshow(grid, text_auto=".0%", aspect="auto", color_continuous_scale="Teal"),
                        width="stretch")
        st.caption("Independent-Poisson baseline on demo data. Not advice; ignores injuries, lineups and form.")

    if fixtures is not None:
        for date, day in fixtures.groupby("date", sort=True):
            st.markdown(f"**{date:%A, %d %b}**")
            for i, row in day.iterrows():
                c1, c2 = st.columns([1, 6])
                c1.write(row["kickoff"])
                if c2.button(f"{row['home_team']} v {row['away_team']}", key=f"fx{i}"):
                    preview(row["home_team"], row["away_team"])

# --------------------------------------------------------------------------- #
with aviator_tab:
    st.subheader("Explore historical crash multipliers")
    uploaded = st.file_uploader("Upload authorized multiplier CSV (column: multiplier)", type=["csv"], key="av")
    try:
        values = load_multipliers(uploaded) if uploaded else sample_multipliers()
    except ValueError as exc:
        st.error(str(exc))
        values = None
    if uploaded is None:
        st.caption("Showing the synthetic sample (500 rounds from a 97%-RTP model).")

    if values is not None:
        summary = summarize_multipliers(values)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rounds in sample", summary["n_rounds"])
        c2.metric("Median multiplier", f'{summary["median"]:.2f}x')
        c3.metric("Minimum", f'{summary["min"]:.2f}x')
        c4.metric("Maximum", f'{summary["max"]:.2f}x')

        st.markdown("#### Observed vs. fair-game model")
        rtp = st.slider("Assumed return-to-player (RTP)", 0.80, 1.00, 0.97, 0.01,
                        help="Share of stakes a memoryless crash game returns on average. 0.97 means a 3% house edge.")
        comp = pd.DataFrame(fair_game_comparison(values, rtp=rtp))
        fig = go.Figure()
        fig.add_bar(x=[f"{t:g}x" for t in comp.threshold], y=comp.observed, name="Observed share",
                    error_y=dict(type="data", symmetric=False, array=comp.ci_high - comp.observed,
                                 arrayminus=comp.observed - comp.ci_low))
        fig.add_scatter(x=[f"{t:g}x" for t in comp.threshold], y=comp.expected, mode="markers",
                        marker=dict(size=12, symbol="diamond"), name="Model: RTP / m")
        fig.update_layout(yaxis=dict(range=[0, 1], title="Share of rounds reaching threshold"),
                          title="Observed share (95% interval) vs. model")
        st.plotly_chart(fig, width="stretch")
        st.caption("Bars show frequencies in this sample, not predictions. Under the model every cash-out target has "
                   "the same expected return: RTP × (RTP / m) × m = RTP, so no target is 'smarter' than another.")

        cashout = st.slider("Cash-out target for return estimate", 1.1, 10.0, 2.0, 0.1, key="rtp_target")
        est = estimate_cashout_rtp(values, cashout)
        lo, hi = est["rtp_interval"]
        e1, e2 = st.columns(2)
        e1.metric("Estimated return per 1.00 staked", f'{est["estimated_rtp"]:.3f}',
                  help=f"95% interval: {lo:.3f} to {hi:.3f}")
        e2.metric("Implied house edge", f'{est["implied_house_edge"]:.1%}')
        st.caption(f"95% interval for return: {lo:.3f} – {hi:.3f}. Wide intervals mean the sample is too small to conclude much.")

        st.markdown("#### Do past rounds carry information about the next?")
        if len(values) >= 10:
            ind = independence_test(values)
            i1, i2 = st.columns(2)
            i1.metric("Lag-1 autocorrelation", f'{ind["lag1_autocorrelation"]:+.3f}')
            i2.metric("Permutation p-value", f'{ind["p_value"]:.2f}')
            st.caption("A large p-value means no evidence that one round predicts the next, "
                       "which is why 'patterns' and 'due' crashes are not usable signals.")
        else:
            st.info("At least 10 rounds are needed for the independence test.")

        st.markdown("#### Hypothetical replay against this historical sample")
        r1, r2, r3 = st.columns(3)
        target = r1.slider("Cash-out threshold (simulation only)", 1.1, 10.0, 2.0, 0.1)
        stake = r2.number_input("Stake per round (hypothetical)", min_value=1.0, value=10.0, step=1.0)
        bankroll = r3.number_input("Starting bankroll (hypothetical)", min_value=0.0, value=1000.0, step=100.0)
        sim = simulate_historical_cashout(values, target, stake, bankroll)
        a, b, c = st.columns(3)
        a.metric("Simulated net result", f"{sim.net_profit_loss:,.2f}")
        b.metric("Winning / losing rounds", f"{sim.winning_rounds} / {sim.losing_rounds}")
        c.metric("Maximum drawdown", f"{sim.max_drawdown:.1%}")
        st.plotly_chart(px.line(y=list(sim.bankroll_path), labels={"x": "Round", "y": "Bankroll"},
                                title="Hypothetical bankroll path"), width="stretch")
        if sim.stopped_early:
            st.warning("Bankroll could no longer cover a stake, so the replay stopped early.")
        st.caption("Historical replay is not a forecast. Simplified rules may differ from the actual game.")

        st.markdown("#### Risk of ruin under an assumed win rate")
        m1, m2, m3 = st.columns(3)
        n_rounds = m1.number_input("Rounds to play", 10, 5000, 200, 10)
        p_win = m2.slider("Assumed win probability", 0.05, 0.95, float(min(0.95, max(0.05, rtp / target))), 0.01,
                          help="Defaults to RTP / cash-out target from the fair-game model.")
        if st.button("Run 2,000 simulated sessions"):
            res = simulate_random_bankroll(int(n_rounds), 2000, p_win, target, stake, bankroll)
            k1, k2, k3 = st.columns(3)
            k1.metric("Median ending bankroll", f'{res["median_ending_bankroll"]:,.0f}')
            k2.metric("Finish below start", f'{res["probability_of_finishing_below_start"]:.0%}')
            k3.metric("Bankroll exhausted", f'{res["probability_of_ruin"]:.0%}')
            st.plotly_chart(px.histogram(x=res["ending_bankrolls"], nbins=40,
                                         labels={"x": "Ending bankroll"}, title="Distribution of session outcomes"),
                            width="stretch")

# --------------------------------------------------------------------------- #
with match_tab:
    st.subheader("Football probability baseline")
    up = st.file_uploader("Upload authorized results CSV (date,home_team,away_team,home_goals,away_goals)",
                          type=["csv"], key="fb")
    try:
        matches = load_matches(up) if up else sample_matches()
    except ValueError as exc:
        st.error(str(exc))
        matches = None
    if up is None:
        st.caption("Included results are synthetic demo data (fictional teams, 60 matches).")

    if matches is not None:
        shrink = st.slider("Shrinkage toward league average", 0.0, 15.0, 4.0, 0.5,
                           help="Higher values distrust small samples more. 0 uses raw averages.")
        model = PoissonMatchModel(shrinkage=shrink).fit(matches)
        teams = sorted(model.known_teams)
        col1, col2 = st.columns(2)
        home = col1.selectbox("Home team", teams, index=0)
        away = col2.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
        if home == away:
            st.info("Choose two different teams.")
        else:
            pred = model.predict_match(home, away)
            p1, p2, p3 = st.columns(3)
            p1.metric("Home win", f'{pred["home_win"]:.1%}')
            p2.metric("Draw", f'{pred["draw"]:.1%}')
            p3.metric("Away win", f'{pred["away_win"]:.1%}')
            st.caption(f'Estimated goals: {home} {pred["home_goal_mean"]:.2f} — {pred["away_goal_mean"]:.2f} {away}')
            matrix = pred["score_matrix"]
            score_df = pd.DataFrame(matrix, index=range(matrix.shape[0]), columns=range(matrix.shape[1]))
            score_df.index.name, score_df.columns.name = "Home goals", "Away goals"
            st.plotly_chart(px.imshow(score_df, labels={"x": "Away goals", "y": "Home goals", "color": "Probability"},
                                      title="Scoreline probability matrix", aspect="auto"), width="stretch")
            st.caption("Baseline only; small samples, injuries, red cards, lineups and many other factors are not captured.")

        st.markdown("#### Backtest: does the model beat a naive baseline?")
        if "date" in matches.columns and len(matches) > 20:
            bt = cached_backtest(matches, shrink)
            lo, hi = bt["brier_skill_interval"]
            b1, b2, b3 = st.columns(3)
            b1.metric("Test matches", bt["n_test_matches"])
            b2.metric("Brier: model vs baseline", f'{bt["brier_model"]:.3f} / {bt["brier_baseline"]:.3f}',
                      help="Lower is better.")
            b3.metric("Brier skill score", f'{bt["brier_skill"]:+.1%}', help=f"95% bootstrap interval: {lo:+.1%} to {hi:+.1%}")
            st.caption(f"Skill > 0 means the model beat always predicting average outcome frequencies. "
                       f"95% bootstrap interval: {lo:+.1%} to {hi:+.1%}. If it spans zero, the sample cannot separate skill from luck.")
            cal = bt["calibration"]
            fig = px.scatter(cal, x="predicted", y="observed", size="count", range_x=[0, 1], range_y=[0, 1],
                             title="Calibration (walk-forward predictions)")
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            st.plotly_chart(fig, width="stretch")
            st.caption("Points on the dashed line mean predicted probabilities match observed frequencies.")
        else:
            st.info("Backtesting needs a 'date' column and more than 20 matches.")

# --------------------------------------------------------------------------- #
with about_tab:
    st.subheader("Responsible-play principles")
    st.markdown("""
- Do not chase losses or borrow money to gamble.
- Set a spending limit before making decisions and stop when it is reached.
- A winning streak does not guarantee another win.
- Treat estimates as uncertain, never as guaranteed outcomes.
- Take a break and seek appropriate support if gambling affects finances, work, relationships, or wellbeing.

**Data and privacy:** this demo does not connect to betting accounts or place bets. Use only data you are authorized to access. Do not upload private or account information.

**Project integrity:** sample records are illustrative/synthetic. Document source, licence, timestamp, coverage, and biases before publishing real data.
""")
    st.markdown("#### Session limit planner")
    budget = st.number_input("Money I can afford to lose entirely", min_value=0.0, value=0.0, step=10.0)
    minutes = st.number_input("Time limit (minutes)", min_value=0, value=30, step=5)
    if budget > 0:
        st.success(f"Write this down before you start: stop at a loss of {budget:,.0f} or after {minutes} minutes, "
                   "whichever comes first. Do not top up after reaching it.")
    st.caption("If gambling is causing harm, contact a local gambling-support service or a trusted person. "
               "Many betting operators also offer self-exclusion and deposit-limit tools.")
