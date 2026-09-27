# Investing for Everyone

A web platform for people who have never invested. You create an account, deposit simulated funds, and a reinforcement-learning agent trades the S&P 500 on your behalf, while a glossary and two explainer pages cover the jargon as you go. I built it as my final-year BSc Computer Science dissertation at the University of Exeter (2025); in the app itself it goes by "AI Investing". Everything is simulated: no real money, no brokerage connection, and nothing here is financial advice.

## Why

Most people in the UK keep their savings in cash and treat the stock market as something for other people. When I asked friends and family why, the answers were almost always one of two: "I don't know what I'm doing" or "I don't have time to keep track of it". This project tries to take both off the table: the agent makes the day-to-day decisions so time is not the barrier, and the glossary and explainers sit on the same site so the dashboard is not foreign to someone seeing a portfolio for the first time.

## What it does

- **Accounts.** Register with a name, email, password and date of birth (with an 18+ check and email validation), log in and out, and change your display name. Passwords are hashed with bcrypt before they are stored.
- **Simulated money.** Deposit and withdraw cash (withdrawals cannot overdraw), and sell some or all of your holdings yourself if you would rather not wait for the agent.
- **Portfolio.** A home dashboard showing your total value (cash plus holdings priced at the current S&P 500 level) and a breakdown page with a cash-versus-shares pie chart, share count and current price.
- **Agent trading.** The Performance page loads the trained PPO model, builds an observation from live market data and your balances, and applies the agent's decision to your portfolio. A background thread also runs the agent once a day at 08:00 for the logged-in user.
- **Market data.** An S&P 500 page with a one-year price chart (matplotlib, rendered through Flet's `MatplotlibChart`), the current level and the one-year high and low.
- **Education.** An About page, a "More Info" section explaining reinforcement learning and the S&P 500, and a glossary of investing and RL terms written for first-timers.

"Shares" in the app are fractional units of the S&P 500 index level itself (`^GSPC`), standing in for an index fund. There are no fees, spreads or dividends.

## The trading agent

The agent is a Proximal Policy Optimisation (PPO) policy trained with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). The training notebook is not part of this repository, so the description below is reconstructed from the saved model's metadata (`flet/agent_model/data` and `system_info.txt`) and from the environment the app imports. Treat it as a description of the shipped model rather than a full training log.

- **Environment.** FinRL's `StockTradingEnv`, a Gymnasium environment, with a single asset: the S&P 500. Each step is one trading day and an episode is one pass over the data, 3,728 trading days or roughly fifteen years of daily history. The price history used for training came from Yahoo Finance and Alpha Vantage.
- **State.** Six raw, unnormalised values: the portfolio balance, the current S&P 500 level, the day's traded volume, the 20-day and 50-day moving averages and a 30-day RSI. The app computes these from the last 50 trading days of Yahoo Finance data in `get_current_data()`.
- **Action.** A single continuous value between -1 and 1 from a Gaussian policy. In training, FinRL scales it to a number of shares to buy or sell. In the app, `update_user_portfolio()` turns it into a trade worth at most 20% of the relevant balance: cash when buying, holdings when selling.
- **Reward.** The change in total portfolio value from one day to the next, scaled (FinRL's default). The agent is rewarded for finishing each day richer, not for any particular pattern of trades.
- **Policy network.** Stable-Baselines3's default `MlpPolicy`: separate actor and critic networks, each 6 → 64 → 64 → 1 with tanh activations, plus a learned log standard deviation for the action.
- **Training run.** 200,000 timesteps (about 54 passes over the data) with 4,096-step rollouts, minibatches of 32, 10 epochs per update, a learning rate of 5e-5, gamma 0.99, GAE lambda 0.95 and no entropy bonus. Trained on CPU in February 2025 under Stable-Baselines3 2.6.0a1, PyTorch 2.5.1 and Python 3.11. `agent.zip` (19 February) is the model the app loads; `ppo_stock_trading.zip` is an earlier run from 16 February with the same settings.
- **How it decides.** On demand from the Performance page, or daily from the scheduler thread, the app loads `agent.zip`, fetches the last 50 days of `^GSPC`, assembles the observation, calls `model.predict(observation, deterministic=True)` and applies the resulting trade to the user's stored balances.

### Limitations

- In backtesting the agent roughly matched buy-and-hold on the S&P 500. It learned a stable, non-random policy and avoided catastrophic positioning, but it found no timing edge, which is about what you would expect from the most heavily analysed index there is. Matching the index is a respectable outcome for new investors; it is not alpha.
- It trades one asset, uses the index level as if it were a tradeable price, and ignores transaction costs.
- Feature parity between training and serving is not enforced in code. FinRL lays the state out as cash, price, shares held and indicators; the app rebuilds the observation from live data with total value and traded volume in the first and third slots. A shared feature module used by both the notebook and the app is the first thing I would add.
- The app is a single-process demo: login state is a module-level variable, so it supports one logged-in user at a time, and the daily scheduler trades only for that user.

## Architecture

Everything runs in one Python process.

```
flet/
├── Dissertation/
│   ├── main.py                 # the whole app: pages, auth, data fetching, agent inference, scheduler
│   ├── agent.zip               # trained PPO model the app loads (Stable-Baselines3 format)
│   ├── ppo_stock_trading.zip   # earlier training run with the same settings
│   ├── graph.py                # generates the decorative background chart, assets/graph1.png
│   ├── assets/                 # images used in the UI
│   └── requirements.txt        # pinned dependencies for deployment
└── agent_model/                # agent.zip unpacked: policy weights, optimiser state, SB3 metadata
```

- **UI:** Flet 0.26, which drives a Flutter front end from Python. `ft.app(..., view=ft.AppView.WEB_BROWSER)` starts Flet's web server and serves the app to a browser. Each page is a Python function that rebuilds `page.controls`; charts are matplotlib figures rendered as SVG through `MatplotlibChart`, plus Flet's built-in `PieChart`.
- **Market data:** Yahoo Finance through `yfinance`. The last 50 trading days of `^GSPC` feed the agent's observation; the last year feeds the chart. No API keys are involved.
- **Agent:** `PPO.load("agent.zip")` from Stable-Baselines3, running on CPU through PyTorch. FinRL provides the environment class the model was trained in.
- **Persistence:** a JSON file, `user_data.json`, keyed by email and holding the bcrypt hash, name, date of birth, cash balance and shares held. It is created on first registration in the directory the app is run from, and it is git-ignored.

## Running locally

Python 3.11 is the safe choice; it is the version the model was saved under.

```bash
git clone https://github.com/henryforrest/investing-for-everyone.git
cd investing-for-everyone/flet/Dissertation
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Flet starts a local web server on a free port and opens the app in your browser. Register an account (the email is only used as a username; date of birth as YYYY-MM-DD), deposit some simulated cash, then open Performance and press "Predict Performance" to have the agent trade.

A few notes:

- No API keys or environment variables are needed. Market data comes from Yahoo Finance, so you do need an internet connection.
- `requirements.txt` was captured from the deployment environment and pins more than the app imports; FinRL in particular pulls in a lot. The direct dependencies are `flet`, `stable-baselines3`, `torch`, `finrl`, `yfinance`, `pandas`, `numpy`, `matplotlib`, `bcrypt` and `schedule`. Expect the first install to take a while.
- Run from `flet/Dissertation` so that the relative paths to `agent.zip`, `assets/` and `user_data.json` resolve.
- To host it, Flet switches to web-server mode automatically on a Linux server without a display (or with `FLET_FORCE_WEB_SERVER=true`). Set `FLET_SERVER_PORT` to pick the port, which defaults to 8000 in that mode.

To check the model without the UI:

```python
from stable_baselines3 import PPO

model = PPO.load("agent.zip")
print(model.observation_space)  # Box(-inf, inf, (6,), float32)
print(model.action_space)       # Box(-1.0, 1.0, (1,), float32)
```

## Status

A demo runs on Render's free tier at https://ai-investing.onrender.com/. The instance sleeps when idle, so the first load can take a minute or two while it wakes; after that it is responsive. It is a demonstration only: no real money changes hands. The project was submitted in 2025 and the code is as it was for the dissertation, plus some later housekeeping (this README, a `.gitignore` and the removal of runtime files).

## Read more

The [case study on my portfolio](https://henryforrest.github.io/dissertation.html) covers the motivation, the design decisions, the honest result on the agent and what I would do differently.
