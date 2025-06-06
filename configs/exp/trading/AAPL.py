root = None
selected_asset = "AAPL.O"
asset_type = "company"
workdir = "workdir/trading"
tool_params_dir = "res/strategy_record/trading"
tool_use_best_params = True
tag = f"{selected_asset}"

initial_amount = 1e7
transaction_cost_pct = 1e-3

# adjust the following parameters mainly
trader_preference = "aggressive_trader"
train_start_date = "2023-03-15"
train_end_date = "2023-07-01"
valid_start_date = "2023-06-01"
valid_end_date = "2024-01-01"

short_term_past_date_range = 1
medium_term_past_date_range = 7
long_term_past_date_range = 28
short_term_next_date_range = 0
medium_term_next_date_range = 0
long_term_next_date_range = 0
look_forward_days = long_term_next_date_range
look_back_days = long_term_past_date_range
previous_action_look_back_days = 14
top_k = 5

train_latest_market_intelligence_summary_template_path = "res/prompts/template/train/trading/latest_market_intelligence_summary.html"
train_past_market_intelligence_summary_template_path = "res/prompts/template/train/trading/past_market_intelligence_summary.html"
train_low_level_reflection_template_path = "res/prompts/template/train/trading/low_level_reflection.html"
train_high_level_reflection_template_path = "res/prompts/template/train/trading/high_level_reflection.html"
train_decision_template_path = "res/prompts/template/train/trading/decision.html"

valid_latest_market_intelligence_summary_template_path = "res/prompts/template/valid/trading/latest_market_intelligence_summary.html"
valid_past_market_intelligence_summary_template_path = "res/prompts/template/valid/trading/past_market_intelligence_summary.html"
valid_low_level_reflection_template_path = "res/prompts/template/valid/trading/low_level_reflection.html"
valid_high_level_reflection_template_path = "res/prompts/template/valid/trading/high_level_reflection.html"
valid_decision_template_path = "res/prompts/template/valid/trading/decision.html"

dataset = dict(
    type="Dataset",
    root=root,
    price_path="datasets/exp_stocks/features",
    news_path="datasets/exp_stocks/news",
    guidance_path=None,
    sentiment_path=None,
    economics_path=None,
    interval="1d",
    assets_path="configs/_asset_list_/exp_stocks.txt",
    workdir=workdir,
    tag=tag
)

train_environment = dict(
    type="EnvironmentTrading",
    mode="train",
    dataset=None,
    selected_asset=selected_asset,
    asset_type=asset_type,
    start_date=train_start_date,
    end_date=train_end_date,
    look_back_days=look_back_days,
    look_forward_days=look_forward_days,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    discount=1.0,
)

valid_environment = dict(
type="EnvironmentTrading",
    mode="train",
    dataset=None,
    selected_asset=selected_asset,
    asset_type=asset_type,
    start_date=valid_start_date,
    end_date=valid_end_date,
    look_back_days=look_back_days,
    look_forward_days=look_forward_days,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    discount=1.0,
)

plots = dict(
    type = "PlotsInterface",
    root = root,
    workdir = workdir,
    tag = tag,
)

memory = dict(
    type="MemoryInterface",
    root=root,
    symbols=None,
    memory_path="memory",
    embedding_dim=None,
    max_recent_steps=5,
    workdir=workdir,
    tag=tag
)

latest_market_intelligence_summary = dict(
    type="LatestMarketIntelligenceSummaryTrading",
    model = "gemini-2.5-pro-exp-03-25"
)

past_market_intelligence_summary = dict(
    type="PastMarketIntelligenceSummaryTrading",
    model = "gemini-2.5-pro-exp-03-25"
)

low_level_reflection = dict(
    type="LowLevelReflectionTrading",
    model = "gemini-2.5-pro-exp-03-25",
    short_term_past_date_range=short_term_past_date_range,
    medium_term_past_date_range=medium_term_past_date_range,
    long_term_past_date_range=long_term_past_date_range,
    short_term_next_date_range=short_term_next_date_range,
    medium_term_next_date_range=medium_term_next_date_range,
    long_term_next_date_range=long_term_next_date_range,
    look_back_days=long_term_past_date_range,
    look_forward_days=long_term_next_date_range
)

high_level_reflection = dict(
    type="HighLevelReflectionTrading",
    model = "gemini-2.5-pro-exp-03-25",
    previous_action_look_back_days=previous_action_look_back_days
)

decision = dict(
    type="DecisionTrading",
    model = "gemini-2.5-pro-exp-03-25",
)

provider = dict(
    type="OpenAIProvider",
    provider_cfg_path="configs/openai_config.json",
)
