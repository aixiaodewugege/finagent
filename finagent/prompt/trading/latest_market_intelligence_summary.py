import math
import os
import backoff
from typing import Dict, List, Any
from copy import deepcopy

from finagent.registry import PROMPT
from finagent.prompt import Prompt
from finagent.asset import ASSET
from finagent.memory import MemoryInterface
from finagent.provider import EmbeddingProvider
from finagent.query import DiverseQuery
from finagent.utils import init_path
from finagent.utils import save_html
from finagent.utils import save_json, load_json

@PROMPT.register_module(force=True)
class LatestMarketIntelligenceSummaryTrading(Prompt):
    def __init__(self,
                 *args,
                 model: Any = None,
                 **kwargs):
        self.model = model
        super(LatestMarketIntelligenceSummaryTrading, self).__init__()

    def convert_to_params(self,
                            state: Dict,
                            info: Dict,
                            params: Dict,
                            memory: MemoryInterface,
                            provider: EmbeddingProvider,
                            diverse_query: DiverseQuery = None) -> Dict:
        res_params = deepcopy(params)

        asset_info = ASSET.get_asset_info(info["symbol"])

        asset_name = asset_info["companyName"]
        asset_symbol = asset_info["symbol"]
        asset_exchange = asset_info["exchange"]
        asset_sector = asset_info["sector"]
        asset_industry = asset_info["industry"]
        asset_description = asset_info["description"]
        asset_type = info["asset_type"]
        current_date = info["date"]

        price = deepcopy(state["price"])
        news = deepcopy(state["news"])

        price = price[price.index == current_date]
        news = news[news.index == current_date]

        if len(news) > 20:
            news = news.sample(n=20)

        latest_market_intelligence_text = f"Date: Today is {current_date}.\n"

        if len(price) > 0:
            open = price["open"].values[0]
            high = price["high"].values[0]
            low = price["low"].values[0]
            close = price["close"].values[0]
            last_close = price["last_close"].values[0]
            volume = price["volume"].values[0]
            PE_ratio = price["PE_ratio"].values[0]
            PE_ratio_ttm = price["PE_ratio_ttm"].values[0]
            PCF_ratio_ttm = price["PCF_ratio_ttm"].values[0]
            PB_ratio = price["PB_ratio"].values[0]
            PS_ratio = price["PS_ratio"].values[0]
            PS_ratio_ttm = price["PS_ratio_ttm"].values[0]

            
            RSI = price["RSI"].values[0]
            ADX = price["ADX"].values[0]
            BB_lower = price["BB_lower"].values[0]
            BB_middle = price["BB_middle"].values[0]
            BB_upper = price["BB_upper"].values[0]
            VWMA = price["VWMA"].values[0]
            CCI = price["CCI"].values[0]
            MACD = price["MACD"].values[0]
            MACD_signal = price["MACD_signal"].values[0]
        
            # 添加基础行情描述
            latest_market_intelligence_text += (
                f"Prices: Open: ({open}), High: ({high}), Low: ({low}), Close: ({close}), "
                f"Last Close: ({last_close}), Volume: ({volume}), PE_ratio: ({PE_ratio}), "
                f"PE_ratio_ttm: ({PE_ratio_ttm}), PCF_ratio_ttm: ({PCF_ratio_ttm}), PB_ratio: ({PB_ratio}), "
                f"PS_ratio: ({PS_ratio}), PS_ratio_ttm: ({PS_ratio_ttm})\n"
            )

            # 添加技术指标分析描述
            technical_text = (
                f"The latest RSI is {RSI:.2f}, indicating the asset is "
                f"{'overbought' if RSI > 70 else 'oversold' if RSI < 30 else 'neutral'}. "
                f"ADX is {ADX:.2f}, suggesting a {'strong' if ADX > 25 else 'weak'} trend. "
                f"The price is {'above' if close > BB_upper else 'below' if close < BB_lower else 'within'} "
                f"the Bollinger Bands (Upper: {BB_upper:.2f}, Middle: {BB_middle:.2f}, Lower: {BB_lower:.2f}). "
                f"VWMA is {VWMA:.2f}, indicating volume-adjusted price momentum. "
                f"CCI is {CCI:.2f}, suggesting {'an overbought signal' if CCI > 100 else 'an oversold signal' if CCI < -100 else 'neutral momentum'}. "
                f"MACD is {MACD:.2f} and Signal Line is {MACD_signal:.2f}, showing a "
                f"{'bullish' if MACD > MACD_signal else 'bearish'} crossover signal.\n"
            )

            latest_market_intelligence_text += technical_text
        else:
            latest_market_intelligence_text += f"Prices: Today is closed for trading.\n"

        if len(news) == 0:
            latest_market_intelligence_text = "There is no latest market_intelligence.\n"
        else:
            latest_market_intelligence_list = []

            for row in news.iterrows():
                row = row[1]
                id = row["id"]
                title = row["title"]
                text = row["text"]

                latest_market_intelligence_item = f"ID: {id}\n" + \
                                                  f"Headline: {title}\n" + \
                                                  f"Content: {text}\n"

                latest_market_intelligence_list.append(latest_market_intelligence_item)

            if len(latest_market_intelligence_list) == 0:
                latest_market_intelligence_text = "There is no latest market_intelligence.\n"
            else:
                latest_market_intelligence_text = "\n".join(latest_market_intelligence_list)

        res_params.update({
            "date": current_date,
            "asset_name": asset_name,
            "asset_type": asset_type,
            "asset_symbol": asset_symbol,
            "asset_exchange": asset_exchange,
            "asset_sector": asset_sector,
            "asset_industry": asset_industry,
            "asset_description": asset_description,
            "latest_market_intelligence": latest_market_intelligence_text,
        })

        return res_params


    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response_dict(self,
                          provider,
                          model,
                          messages,
                          check_keys: List[str] = None):

        check_keys = [
            "query",
            "summary"
        ]

        response_dict, res_html = super(LatestMarketIntelligenceSummaryTrading, self).get_response_dict(provider=provider,
                                                                                        model=model,
                                                                                        messages=messages,
                                                                                        check_keys=check_keys)

        return response_dict, res_html

    def add_to_memory(self,
                      state: Dict,
                      info: Dict,
                      res: Dict,
                      memory: MemoryInterface = None,
                      provider: EmbeddingProvider = None) -> None:

        response_dict = deepcopy(res["response_dict"])

        current_date = info["date"]
        stock_symbol = info["symbol"]
        
        price = deepcopy(state["price"])
        news = deepcopy(state["news"])

        price = price[price.index == current_date]
        news = news[news.index == current_date]

        if len(price) > 0:
            open = price["open"].values[0]
            high = price["high"].values[0]
            low = price["low"].values[0]
            close = price["close"].values[0]
            last_close = price["last_close"].values[0]
            volume = price["volume"].values[0]
            PE_ratio = price["PE_ratio"].values[0]
            PE_ratio_ttm = price["PE_ratio_ttm"].values[0]
            PCF_ratio_ttm = price["PCF_ratio_ttm"].values[0]
            PB_ratio = price["PB_ratio"].values[0]
            PS_ratio = price["PS_ratio"].values[0]
            PS_ratio_ttm = price["PS_ratio_ttm"].values[0]
        else:
            open = math.nan
            high = math.nan
            low = math.nan
            close = math.nan

        for row in news.iterrows():
            date = row[0] if isinstance(row[0], str) else row[0].strftime("%Y-%m-%d")
            row = row[1]

            id = row["id"]
            title = row["title"]
            text = row["text"]

            embedding_text = f"Heading: {title}\n" + \
                             f"Content: {text}\n"

            embedding = provider.embed_query(embedding_text)

            data = {
                "date": date,
                "id": id,
                "title": title,
                "text": text,
                "open": open,
                "high": high,
                "low": low,
                "close": close,
                "last_close": last_close,
                "volume": volume,
                "PE_ratio": PE_ratio,
                "PE_ratio_ttm": PE_ratio_ttm,
                "PCF_ratio_ttm": PCF_ratio_ttm,
                "PB_ratio": PB_ratio,
                "PS_ratio": PS_ratio,
                "PS_ratio_ttm": PS_ratio_ttm,
                "query": response_dict["query"],
                "summary": response_dict["summary"],
                "embedding_text": embedding_text,
                "embedding": embedding,
            }

            memory.add_memory(type="market_intelligence",
                              symbol=stock_symbol,
                              data=data,
                              embedding_key="embedding")

    def run(self,
            state: Dict,
            info: Dict,
            params: Dict,
            template: Any = None,
            memory: MemoryInterface = None,
            provider: EmbeddingProvider = None,
            diverse_query: DiverseQuery = None,
            exp_path: str = None,
            save_dir: str = None,
            call_provider = True,
            **kwargs):

        print(">" * 50 + f"{info['date']} - Running Latest Market Intelligence Summary Trading Prompt" + ">" * 50)

        # init path
        res_json_path = init_path(os.path.join(exp_path, "json", save_dir, "latest_market_intelligence"))
        html_path = init_path(os.path.join(exp_path, "html", save_dir, "latest_market_intelligence"))

        if call_provider:
            # summary latest market intelligence
            task_params = self.convert_to_params(state=state,
                                                 info=info,
                                                 params=params,
                                                 memory=memory,
                                                 provider=provider,
                                                 diverse_query=diverse_query)
            message, html = self.to_message(params=task_params, template=template)
            # response_dict, res_html = self.get_response_dict(provider = provider,
            #                                               model=self.model,
            #                                               messages=message)

            # query = response_dict["query"]
            # summary = response_dict["summary"]
            max_retries = 10
            REQUIRED_KEYS = ["query", "summary"]

            for attempt in range(max_retries):
                print(f"[Attempt {attempt + 1}/{max_retries}] Calling LLM provider...")

                response_dict, res_html = self.get_response_dict(
                    provider=provider,
                    model=self.model,
                    messages=message
                )

                try:
                    query = response_dict["query"]
                    summary = response_dict["summary"]
                    break  # ✅ 成功提取，退出循环

                except KeyError as e:
                    print(f"[WARN] Missing key: {str(e)}")
                    if attempt < max_retries - 1:
                        print("[INFO] Retrying LLM call...")
                    else:
                        print("[ERROR] Max retries reached. Using fallback values.")
                        query = response_dict.get("query", "[MISSING QUERY]")
                        summary = response_dict.get("summary", "[MISSING SUMMARY]")


            html = html.prettify()
            res_html = res_html.prettify()

            res = {
                "params": task_params,
                "message": message,
                "html": html,
                "res_html": res_html,
                "response_dict": response_dict,
            }

        else:

            res = load_json(os.path.join(res_json_path, f"res_{info['date']}.json"))

            task_params= res["params"]

            html = res["html"]
            res_html = res["res_html"]
            query = res["response_dict"]["query"]
            summary = res["response_dict"]["summary"]

        params.update(task_params)

        params.update({
            "latest_market_intelligence_query": query,
            "latest_market_intelligence_summary": summary,
        })

        save_html(html, os.path.join(html_path, f"prompt_{info['date']}.html"))
        save_html(res_html, os.path.join(html_path, f"res_{info['date']}.html"))
        save_json(res, os.path.join(res_json_path, f"res_{info['date']}.json"))

        print("<" * 50 + f"{info['date']} - Finish Running Latest Market Intelligence Summary Trading Prompt" + "<" * 50)

        return res
