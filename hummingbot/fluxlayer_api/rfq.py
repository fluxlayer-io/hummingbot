from hummingbot.logger import HummingbotLogger
from hummingbot.fluxlayer_api.get_chain_gas import get_btc_fee, get_gas_prices, get_solana_fee
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.data_type.order_book_tracker import OrderBookTracker
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.connector.exchange.okx.okx_exchange import OkxExchange
from hummingbot.connector.exchange.okx.okx_api_order_book_data_source import OkxAPIOrderBookDataSource
from hummingbot.connector.exchange.okx import okx_constants as OKX_CONSTANTS
from hummingbot.connector.exchange.mexc.mexc_exchange import MexcExchange
from hummingbot.connector.exchange.mexc.mexc_api_order_book_data_source import MexcAPIOrderBookDataSource
from hummingbot.connector.exchange.mexc import mexc_constants as MEXC_CONSTANTS
from hummingbot.connector.exchange.kucoin.kucoin_exchange import KucoinExchange
from hummingbot.connector.exchange.kucoin.kucoin_api_order_book_data_source import KucoinAPIOrderBookDataSource
from hummingbot.connector.exchange.kucoin import kucoin_constants as KUCOIN_CONSTANTS
from hummingbot.connector.exchange.gate_io.gate_io_exchange import GateIoExchange
from hummingbot.connector.exchange.gate_io.gate_io_api_order_book_data_source import GateIoAPIOrderBookDataSource
from hummingbot.connector.exchange.gate_io import gate_io_constants as GATE_IO_CONSTANTS
from hummingbot.connector.exchange.bybit.bybit_exchange import BybitExchange
from hummingbot.connector.exchange.bybit.bybit_api_order_book_data_source import BybitAPIOrderBookDataSource
from hummingbot.connector.exchange.bybit import bybit_constants as BYBIT_CONSTANTS
from hummingbot.connector.exchange.bitmart.bitmart_exchange import BitmartExchange
from hummingbot.connector.exchange.bitmart.bitmart_api_order_book_data_source import BitmartAPIOrderBookDataSource
from hummingbot.connector.exchange.bitmart import bitmart_constants as BITMART_CONSTANTS
from hummingbot.connector.exchange.bing_x.bing_x_exchange import BingXExchange
from hummingbot.connector.exchange.bing_x.bing_x_api_order_book_data_source import BingXAPIOrderBookDataSource
from hummingbot.connector.exchange.bing_x import bing_x_constants as BING_X_CONSTANTS
from hummingbot.connector.exchange.binance.binance_exchange import BinanceExchange
from hummingbot.connector.exchange.binance.binance_api_order_book_data_source import BinanceAPIOrderBookDataSource
from hummingbot.connector.exchange.binance import binance_constants as BINANCE_CONSTANTS
from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.client.config.client_config_map import AnonymizedMetricsEnabledMode, ClientConfigMap
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from decimal import Decimal

import aiohttp
import asyncpg
import requests
from aiohttp import ClientSession, ClientTimeout, TCPConnector

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

# 配置日志
_logger = HummingbotLogger(__name__)

# 交易所配置
EXCHANGES = {
    "binance": {
        "exchange_class": BinanceExchange,
        "data_source_class": BinanceAPIOrderBookDataSource,
        "constants": BINANCE_CONSTANTS,
        "proxy": "http://127.0.0.1:7897",
        "required_params": {
            "binance_api_key": "",
            "binance_api_secret": ""
        }
    },
    "gate_io": {
        "exchange_class": GateIoExchange,
        "data_source_class": GateIoAPIOrderBookDataSource,
        "constants": GATE_IO_CONSTANTS,
        "proxy": "http://127.0.0.1:7897",
        "required_params": {
            "gate_io_api_key": "",
            "gate_io_secret_key": ""
        }
    },
    "bybit": {
        "exchange_class": BybitExchange,
        "data_source_class": BybitAPIOrderBookDataSource,
        "constants": BYBIT_CONSTANTS,
        "proxy": "http://127.0.0.1:7897",
        "required_params": {
            "bybit_api_key": "",
            "bybit_api_secret": ""
        }
    },
    "bing_x": {
        "exchange_class": BingXExchange,
        "data_source_class": BingXAPIOrderBookDataSource,
        "constants": BING_X_CONSTANTS,
        "proxy": "http://127.0.0.1:7897",
        "required_params": {
            "bingx_api_key": "",
            "bingx_api_secret": ""
        }
    },
    "kucoin": {
        "exchange_class": KucoinExchange,
        "data_source_class": KucoinAPIOrderBookDataSource,
        "constants": KUCOIN_CONSTANTS,
        "proxy": "http://127.0.0.1:7897",
        "required_params": {
            "kucoin_api_key": "",
            "kucoin_passphrase": "",
            "kucoin_secret_key": ""
        }
    },
    "bitmart": {
        "exchange_class": BitmartExchange,
        "data_source_class": BitmartAPIOrderBookDataSource,
        "constants": BITMART_CONSTANTS,
        "proxy": "http://127.0.0.1:7897",
        "required_params": {
            "bitmart_api_key": "",
            "bitmart_secret_key": "",
            "bitmart_memo": "",
        }
    },
    "okx": {
        "exchange_class": OkxExchange,
        "data_source_class": OkxAPIOrderBookDataSource,
        "constants": OKX_CONSTANTS,
        "proxy": "http://127.0.0.1:7897",
        "required_params": {
            "okx_api_key": "",
            "okx_secret_key": "",
            "okx_passphrase": "",
        }
    },
    "mexc": {
        "exchange_class": MexcExchange,
        "data_source_class": MexcAPIOrderBookDataSource,
        "constants": MEXC_CONSTANTS,
        "proxy": "http://127.0.0.1:7897",
        "required_params": {
            "mexc_api_key": "",
            "mexc_api_secret": ""
        }
    }
}

CHAIN_GAS_TOKEN_MAP = {
    "Ethereum": "ETH",
    "BSC": "BNB",
    "Polygon": "MATIC",
    "Arbitrum": "ETH",
    "BTC": "BTC",
    "Solana": "SOL",
}

# 全局变量
_exchange = None
_order_book_tracker = None
_initialized = False
_throttler = None
_initialization_lock = asyncio.Lock()
_session = None


async def initialize_exchange(exchange_name: str = "binance", trading_pairs: list = None):
    """初始化交易所连接"""
    global _exchange, _order_book_tracker, _initialized, _throttler, _session

    if exchange_name not in EXCHANGES:
        raise ValueError(f"Unsupported exchange: {exchange_name}")

    exchange_config = EXCHANGES[exchange_name]

    async with _initialization_lock:
        if not _initialized:
            try:
                # 创建客户端配置
                client_config = ClientConfigMap()
                client_config.anonymized_metrics_mode = AnonymizedMetricsEnabledMode()
                client_config_adapter = ClientConfigAdapter(client_config)

                # 创建交易所实例
                exchange_params = {
                    "client_config_map": client_config_adapter,
                    "trading_pairs": trading_pairs or ["BTC-USDT"],
                    "trading_required": False
                }
                # 添加交易所特定的必要参数
                exchange_params.update(exchange_config["required_params"])

                _exchange = exchange_config["exchange_class"](**exchange_params)

                # 创建 throttler
                _throttler = AsyncThrottler(exchange_config["constants"].RATE_LIMITS)

                # 创建带代理的 ClientSession
                connector = TCPConnector(ssl=False)
                _session = ClientSession(
                    connector=connector,
                    timeout=ClientTimeout(total=30),
                    trust_env=True,
                    proxy=exchange_config["proxy"]
                )

                # 创建 WebAssistantsFactory
                api_factory = WebAssistantsFactory(
                    throttler=_throttler
                )

                # 创建订单簿数据源
                data_source = exchange_config["data_source_class"](
                    trading_pairs=trading_pairs or ["BTC-USDT"],
                    connector=_exchange,
                    api_factory=api_factory
                )

                # 创建订单簿跟踪器
                _order_book_tracker = OrderBookTracker(
                    data_source=data_source,
                    trading_pairs=trading_pairs or ["BTC-USDT"]
                )

                # 启动订单簿跟踪器
                _order_book_tracker.start()

                # 等待订单簿数据加载
                await _order_book_tracker.wait_ready()

                _initialized = True
                _logger.info(f"Successfully initialized {exchange_name} exchange")
            except Exception as e:
                _logger.error(f"Failed to initialize exchange: {e}")
                raise


async def cleanup():
    """清理资源"""
    global _order_book_tracker, _initialized, _throttler, _session

    try:
        if _order_book_tracker is not None:
            _order_book_tracker.stop()
        if _session is not None:
            await _session.close()
        _initialized = False
        _logger.info("Successfully cleaned up resources")
    except Exception as e:
        _logger.error(f"Error during cleanup: {e}")
        raise


def calculate_price_impact(order_book_data, amount, is_buy=True):
    """
    计算指定交易量对价格的影响

    参数:
        order_book_data: 订单簿数据 
            - 买入时传入 asks [[price, qty], ...]
            - 卖出时传入 bids [[price, qty], ...]
        amount: 交易数量
            - 买入时为 USDT 预算
            - 卖出时为要卖的代币数量
        is_buy: True=买入, False=卖出

    返回:
        dict: 包含价格影响分析的结果
    """
    if order_book_data is None or len(order_book_data) == 0:
        return {
            'final_price': None,
            'total_amount': 0,
            'average_price': None,
            'price_impact': 0
        }

    if is_buy:
        # 买入逻辑：用 USDT 预算买代币，消费 asks
        return _calculate_buy_impact(order_book_data, amount)
    else:
        # 卖出逻辑：卖代币换 USDT，消费 bids
        return _calculate_sell_impact(order_book_data, amount)


def _calculate_buy_impact(asks, budget_usdt):
    """计算买入的价格影响"""
    remaining_budget = budget_usdt
    total_amount = 0  # 买到的代币数量
    last_price = None
    executed_orders = []

    for price_str, qty_str in asks:
        if remaining_budget <= 0:
            break

        price = float(price_str)
        qty = float(qty_str)
        order_value = price * qty

        if order_value <= remaining_budget:
            # 全量吃单
            remaining_budget -= order_value
            total_amount += qty
            executed_orders.append((price, qty))
            last_price = price
        else:
            # 部分吃单
            executable_qty = remaining_budget / price
            total_amount += executable_qty
            executed_orders.append((price, executable_qty))
            remaining_budget = 0
            last_price = price

    return {
        'final_price': last_price,
        'total_amount': total_amount,  # 买到的代币数量
        'average_price': budget_usdt / total_amount if total_amount > 0 else None,
        'price_impact': (last_price - float(asks[0][0])) / float(asks[0][0]) * 100 if last_price is not None else 0
    }


def _calculate_sell_impact(bids, token_amount):
    """计算卖出的价格影响"""
    remaining_amount = token_amount  # 剩余要卖的代币数量
    total_usdt = 0  # 获得的 USDT 总额
    last_price = None
    executed_orders = []

    for price_str, qty_str in bids:
        if remaining_amount <= 0:
            break

        price = float(price_str)
        qty = float(qty_str)

        if qty <= remaining_amount:
            # 全量吃单
            remaining_amount -= qty
            total_usdt += price * qty
            executed_orders.append((price, qty))
            last_price = price
        else:
            # 部分吃单
            executable_qty = remaining_amount
            total_usdt += price * executable_qty
            executed_orders.append((price, executable_qty))
            remaining_amount = 0
            last_price = price

    sold_amount = token_amount - remaining_amount

    return {
        'final_price': last_price,
        'total_amount': total_usdt,  # 获得的 USDT 总额
        'average_price': total_usdt / sold_amount if sold_amount > 0 else None,  # 平均卖价
        'price_impact': (float(bids[0][0]) - last_price) / float(bids[0][0]) * 100 if last_price is not None else 0
    }


def parse_chain_token(chain_token: str):
    """解析链和代币信息"""
    try:
        chain, token = chain_token.split('_')
        return chain, token
    except:
        raise ValueError(f"Invalid chain_token format: {chain_token}")


async def get_order_book(exchange_name: str, token: str):
    """获取代币的USDT价格"""
    symbol = f"{token}-USDT"
    try:
        # 确保交易所已初始化
        # TODO 这里的交易所初始化可能需要改进
        await initialize_exchange(exchange_name, [symbol])

        # 获取订单簿数据
        order_book = _order_book_tracker.order_books[symbol]

        # 获取买卖单
        bids = [(str(price), str(amount)) for price, amount, _ in order_book.bid_entries()]
        asks = [(str(price), str(amount)) for price, amount, _ in order_book.ask_entries()]
        return {
            'lastUpdateId': int(time.time() * 1000),
            'bids': bids,
            'asks': asks
        }
    except Exception as e:
        _logger.error(f"Error getting price for {token}: {e}")
        return None


async def get_quote_price(exchange_name, trading_pair: str, is_buy: bool, quote_volume):
    """获取交易对的报价"""
    try:
        # TODO 这里初始化交易所的方式可能需要改进
        await initialize_exchange(exchange_name, [trading_pair])
        order_book = _order_book_tracker.order_books[trading_pair]
        price = order_book.get_price_for_quote_volume(is_buy, quote_volume)
        return price
    except Exception as e:
        _logger.error(f"Error getting quote price for {trading_pair}: {e}")
        return None


def calculate_gas_fee(chain: str):
    """计算链上的gas费用"""
    if chain.upper() == "BTC":
        btc_fee = get_btc_fee()
        return "BTC", btc_fee["regular"] * 0.00000001  # 转换为BTC
    elif chain == "Solana":
        # {"gasPrice": 0.5, "gasPriceToken": "SOL", "gasLimit": 200000, "gasCost": 0.000105}
        data = get_solana_fee()
        return "SOL", data["gasCost"]
    else:
        gas_prices = get_gas_prices(chain)
        return "ETH", gas_prices["base_fee"] * 21000 / 1e9


async def get_single_exchange_rfq(
    source_chain: str,
    source_token: str,
    amount: float,
    target_chain: str,
    target_token: str,
    exchange_name: str = "binance",
    is_buy: bool = True
):
    """获取单个交易所的RFQ结果"""
    try:
        # 参数验证
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")

        source_orderbook = await get_order_book(exchange_name, source_token)
        if not source_orderbook:
            return None

        # 其他逻辑保持不变...
        if is_buy:
            src_market_price = float(source_orderbook['asks'][0][0])
        else:
            src_market_price = float(source_orderbook['bids'][0][0])

        if not src_market_price:
            return None

        target_orderbook = await get_order_book(exchange_name, target_token)
        if not target_orderbook:
            return None

        if is_buy:
            target_market_price = float(target_orderbook['bids'][0][0])
        else:
            target_market_price = float(target_orderbook['asks'][0][0])

        if not target_market_price:
            return None

        # 计算gas费用
        gas_token, gas_amount = calculate_gas_fee(target_chain)
        # 获取目标链的gas费用
        gas_price_orderbook = await get_order_book(exchange_name, gas_token)
        gas_price = gas_price_orderbook['asks'][0][0]
        gas_price = float(gas_price)  # 转换为 float

        tar_gas_fee_usdt = gas_amount * gas_price
        # 计算扣除gas费用后的USDT价值
        net_value_usdt = amount - tar_gas_fee_usdt

        # 计算价格影响
        if is_buy:
            # 买入时使用 asks
            order_data = source_orderbook['asks']
        else:
            # 卖出时使用 bids
            order_data = source_orderbook['bids']

        analysis = calculate_price_impact(order_data, net_value_usdt, is_buy)
        if analysis is None:
            return None

        average_price = analysis['average_price']
        average_price = average_price * 1.001  # 0.1% 交易所手续费
        price_impact = analysis['price_impact']
        if price_impact * 10000 > 10:  # fluxlayer 抽成
            fluxlayer_price = average_price * 1.01
        else:
            fluxlayer_price = average_price * 1.004
        # 计算target代币数量
        target_amount = net_value_usdt / fluxlayer_price
        # 统一使用 target_amount 作为键名
        return {
            "exchange": exchange_name,
            "source_amount": amount,
            "source_price": src_market_price,
            "target_amount": target_amount,
            "target_price": fluxlayer_price,
        }
    except Exception as e:
        _logger.error(f"Error in {exchange_name} RFQ: {e}")
        return None
    finally:
        await cleanup()


async def get_best_rfq(
    source_chain: str,
    source_token: str,
    amount: float,
    target_chain: str,
    target_token: str,
    is_buy: bool
):
    """并行查询多个交易所并返回最优报价"""
    try:
        # 参数验证
        if amount <= 0:
            return {"error": "Amount must be greater than 0"}
        if not all([source_chain, source_token, target_chain, target_token]):
            return {"error": "Missing required parameters"}

        tasks = [
            get_single_exchange_rfq(
                source_chain, source_token, amount,
                target_chain, target_token, exchange, is_buy
            )
            for exchange in EXCHANGES.keys()
        ]

        results = await asyncio.gather(*tasks)
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return {"error": "No valid quotes available"}

        best_result = max(valid_results, key=lambda x: x["target_amount"])

        # 统一返回格式
        best_result["all_exchanges"] = {
            r["exchange"]: {
                "target_amount": r["target_amount"],
                "target_price": r["target_price"]
            } for r in valid_results
        }

        return best_result
    except Exception as e:
        _logger.error(f"Error in get_best_rfq: {e}")
        return {"error": str(e)}


async def get_generic_rfq_request(
    source_chain: str,
    source_token: str,
    amount: float,
    target_chain: str,
    target_token: str,
    is_buy: bool
):
    """
    从所有支持的Solvers中获取RFQ报价并返回最优价格

    参数:
        source_chain (str): 源链
        source_token (str): 源代币
        amount (float): 源代币数量
        target_chain (str): 目标链
        target_token (str): 目标代币
        is_buy (bool): 是否为买入操作

    返回:
        dict: 包含最优价格的RFQ结果
    """
    try:
        # 从数据库中查询支持指定网络和资产的solvers
        async def get_supported_solvers():
            query = """
                    SELECT id, name, api_endpoint
                    FROM solvers
                    WHERE $1 = ANY (supported_network)
                        AND $2 = ANY (supported_network)
                        AND (supported_asset -> $1) ? $3
                        AND (supported_asset -> $2) ? $4
                    """
            _logger.error("Executing SQL:\n%s\nWith params: %s, %s, %s, %s",
                          query.strip(), source_chain, target_chain, source_token, target_token)

            conn = await get_db_connection()
            return await conn.fetch(
                query,
                source_chain,
                target_chain,
                source_token,
                target_token
            )

        # 向单个solver请求RFQ
        async def get_solver_rfq(solver):
            try:
                timeout = ClientTimeout(total=60)  # 5秒超时
                async with ClientSession(timeout=timeout) as session:
                    payload = {
                        "source_chain": source_chain,
                        "source_token": source_token,
                        "amount": str(amount),
                        "target_chain": target_chain,
                        "target_token": target_token,
                        "is_buy": is_buy,
                    }
                    async with session.post(solver['api_endpoint'], json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            result['solver_name'] = solver['name']
                            return result
                        return None
            except Exception as e:
                _logger.error(f"Error getting RFQ from solver {solver['name']}: {e}")
                return None

        # 获取所有支持的solvers
        solvers = await get_supported_solvers()

        if not solvers:
            return {"error": "No supported solvers found"}

        # 并行请求所有solver的RFQ
        tasks = [get_solver_rfq(solver) for solver in solvers]
        results = await asyncio.gather(*tasks)

        # 过滤掉失败的结果
        valid_results = [r for r in results if r is not None and 'target_amount' in r]

        if not valid_results:
            return {"error": "Failed to get valid RFQ from any solver"}

        # 找出target_amount最大的结果
        best_result = max(valid_results, key=lambda x: float(x['target_amount']))

        # 添加所有solvers的报价信息
        best_result["all_solvers"] = {
            r['solver_name']: {
                "target_amount": r["target_amount"],
                "target_price": r["target_price"]
            } for r in valid_results
        }

        return best_result

    except Exception as e:
        _logger.error(f"Error in generic RFQ request: {e}")
        return {"error": str(e)}


async def get_db_connection():
    """获取 PostgreSQL 异步数据库连接"""
    return await asyncpg.connect(
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "postgres"),
        database=os.getenv("PG_DATABASE", "postgres"),
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 5432)),
    )
