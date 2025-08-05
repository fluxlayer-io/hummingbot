import asyncio
import json
import os
import sys
import time
from datetime import datetime
from decimal import ROUND_DOWN, Decimal, getcontext

import aiohttp
import requests
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from hummingbot.client.config.client_config_map import AnonymizedMetricsEnabledMode, ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
# from hummingbot.client.hummingbot_application import HummingbotApplication  # 暂时注释掉，避免循环导入
from hummingbot.connector.exchange.binance import binance_constants as BINANCE_CONSTANTS
from hummingbot.connector.exchange.binance.binance_api_order_book_data_source import BinanceAPIOrderBookDataSource
from hummingbot.connector.exchange.binance.binance_exchange import BinanceExchange
from hummingbot.connector.exchange.bing_x import bing_x_constants as BING_X_CONSTANTS
from hummingbot.connector.exchange.bing_x.bing_x_api_order_book_data_source import BingXAPIOrderBookDataSource
from hummingbot.connector.exchange.bing_x.bing_x_exchange import BingXExchange
from hummingbot.connector.exchange.bitmart import bitmart_constants as BITMART_CONSTANTS
from hummingbot.connector.exchange.bitmart.bitmart_api_order_book_data_source import BitmartAPIOrderBookDataSource
from hummingbot.connector.exchange.bitmart.bitmart_exchange import BitmartExchange
from hummingbot.connector.exchange.bybit import bybit_constants as BYBIT_CONSTANTS
from hummingbot.connector.exchange.bybit.bybit_api_order_book_data_source import BybitAPIOrderBookDataSource
from hummingbot.connector.exchange.bybit.bybit_exchange import BybitExchange
from hummingbot.connector.exchange.gate_io import gate_io_constants as GATE_IO_CONSTANTS
from hummingbot.connector.exchange.gate_io.gate_io_api_order_book_data_source import GateIoAPIOrderBookDataSource
from hummingbot.connector.exchange.gate_io.gate_io_exchange import GateIoExchange
from hummingbot.connector.exchange.kucoin import kucoin_constants as KUCOIN_CONSTANTS
from hummingbot.connector.exchange.kucoin.kucoin_api_order_book_data_source import KucoinAPIOrderBookDataSource
from hummingbot.connector.exchange.kucoin.kucoin_exchange import KucoinExchange
from hummingbot.connector.exchange.mexc import mexc_constants as MEXC_CONSTANTS
from hummingbot.connector.exchange.mexc.mexc_api_order_book_data_source import MexcAPIOrderBookDataSource
from hummingbot.connector.exchange.mexc.mexc_exchange import MexcExchange
from hummingbot.connector.exchange.okx import okx_constants as OKX_CONSTANTS
from hummingbot.connector.exchange.okx.okx_api_order_book_data_source import OkxAPIOrderBookDataSource
from hummingbot.connector.exchange.okx.okx_exchange import OkxExchange
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.data_type.order_book_tracker import OrderBookTracker
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.fluxlayer_api.get_chain_gas import get_btc_fee, get_gas_prices, get_solana_fee
from hummingbot.logger import HummingbotLogger
from hummingbot.connector.exchange.hyperliquid import hyperliquid_constants as HYPERLIQUID_CONSTANTS
from hummingbot.connector.exchange.hyperliquid.hyperliquid_api_order_book_data_source import HyperliquidAPIOrderBookDataSource
from hummingbot.connector.exchange.hyperliquid.hyperliquid_exchange import HyperliquidExchange

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
        "required_params": {
            "binance_api_key": "",
            "binance_api_secret": ""
        }
    },
    "gate_io": {
        "exchange_class": GateIoExchange,
        "data_source_class": GateIoAPIOrderBookDataSource,
        "constants": GATE_IO_CONSTANTS,
        "required_params": {
            "gate_io_api_key": "",
            "gate_io_secret_key": ""
        }
    },
    "bing_x": {
        "exchange_class": BingXExchange,
        "data_source_class": BingXAPIOrderBookDataSource,
        "constants": BING_X_CONSTANTS,
        "required_params": {
            "bingx_api_key": "",
            "bingx_api_secret": ""
        }
    },
    "kucoin": {
        "exchange_class": KucoinExchange,
        "data_source_class": KucoinAPIOrderBookDataSource,
        "constants": KUCOIN_CONSTANTS,
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
        "required_params": {
            "mexc_api_key": "",
            "mexc_api_secret": ""
        }
    },
    "hyperliquid": {
        "exchange_class": HyperliquidExchange,
        "data_source_class": HyperliquidAPIOrderBookDataSource,
        "constants": HYPERLIQUID_CONSTANTS,
        "required_params": {
            "hyperliquid_api_key": "",
            "hyperliquid_api_secret": "",
            "use_vault": False
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

# 连接器缓存
_connectors = {}
_connector_locks = {}


def load_api_keys_from_env():
    """从环境变量加载API密钥"""
    try:
        # 尝试加载 .env 文件
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
        print("Continuing without .env file support...")
    
    # Binance
    EXCHANGES["binance"]["required_params"]["binance_api_key"] = os.getenv("BINANCE_API_KEY", "")
    EXCHANGES["binance"]["required_params"]["binance_api_secret"] = os.getenv("BINANCE_API_SECRET", "")
    
    # Gate.io
    EXCHANGES["gate_io"]["required_params"]["gate_io_api_key"] = os.getenv("GATE_IO_API_KEY", "")
    EXCHANGES["gate_io"]["required_params"]["gate_io_secret_key"] = os.getenv("GATE_IO_SECRET_KEY", "")
    
    # BingX
    EXCHANGES["bing_x"]["required_params"]["bingx_api_key"] = os.getenv("BINGX_API_KEY", "")
    EXCHANGES["bing_x"]["required_params"]["bingx_api_secret"] = os.getenv("BINGX_API_SECRET", "")
    
    # KuCoin
    EXCHANGES["kucoin"]["required_params"]["kucoin_api_key"] = os.getenv("KUCOIN_API_KEY", "")
    EXCHANGES["kucoin"]["required_params"]["kucoin_secret_key"] = os.getenv("KUCOIN_SECRET_KEY", "")
    EXCHANGES["kucoin"]["required_params"]["kucoin_passphrase"] = os.getenv("KUCOIN_PASSPHRASE", "")
    
    # BitMart
    EXCHANGES["bitmart"]["required_params"]["bitmart_api_key"] = os.getenv("BITMART_API_KEY", "")
    EXCHANGES["bitmart"]["required_params"]["bitmart_secret_key"] = os.getenv("BITMART_SECRET_KEY", "")
    EXCHANGES["bitmart"]["required_params"]["bitmart_memo"] = os.getenv("BITMART_MEMO", "")
    
    # OKX
    EXCHANGES["okx"]["required_params"]["okx_api_key"] = os.getenv("OKX_API_KEY", "")
    EXCHANGES["okx"]["required_params"]["okx_secret_key"] = os.getenv("OKX_SECRET_KEY", "")
    EXCHANGES["okx"]["required_params"]["okx_passphrase"] = os.getenv("OKX_PASSPHRASE", "")
    
    # MEXC
    EXCHANGES["mexc"]["required_params"]["mexc_api_key"] = os.getenv("MEXC_API_KEY", "")
    EXCHANGES["mexc"]["required_params"]["mexc_api_secret"] = os.getenv("MEXC_API_SECRET", "")
    
    # Hyperliquid
    EXCHANGES["hyperliquid"]["required_params"]["hyperliquid_api_key"] = os.getenv("HYPERLIQUID_API_KEY", "")
    EXCHANGES["hyperliquid"]["required_params"]["hyperliquid_api_secret"] = os.getenv("HYPERLIQUID_API_SECRET", "")
    EXCHANGES["hyperliquid"]["required_params"]["use_vault"] = os.getenv("HYPERLIQUID_USE_VAULT", "False").lower() == "true"


# 自动从环境变量加载API密钥
load_api_keys_from_env()


async def get_or_create_connector(connector_name: str, trading_pair: str, wait_for_orderbook: bool = True):
    """
    获取或创建连接器实例（使用缓存）
    
    参数:
        connector_name (str): 交易所名称
        trading_pair (str): 交易对
        wait_for_orderbook (bool): 是否等待订单簿数据初始化完成
        
    返回:
        连接器实例
    """
    # 为每个交易所创建独立的锁
    if connector_name not in _connector_locks:
        _connector_locks[connector_name] = asyncio.Lock()
    
    async with _connector_locks[connector_name]:
        # 检查是否已有缓存的连接器
        if connector_name in _connectors:
            connector = _connectors[connector_name]
            # 检查连接器是否仍然有效且数据是最新的
            try:
                # 检查连接器状态和订单簿数据
                if (hasattr(connector, 'ready') and connector.ready and
                    hasattr(connector, 'order_book_tracker') and 
                    connector.order_book_tracker.ready and
                    trading_pair in connector.order_book_tracker.order_books):
                    
                    # 检查订单簿数据是否是最新的（避免使用过期数据）
                    order_book = connector.order_book_tracker.order_books[trading_pair]
                    if order_book:
                        try:
                            # 使用迭代器方式检查是否有数据，避免len()的问题
                            bid_iter = order_book.bid_entries()
                            ask_iter = order_book.ask_entries()
                            
                            # 检查是否至少有一个条目
                            has_bids = False
                            has_asks = False
                            try:
                                next(bid_iter)
                                has_bids = True
                            except StopIteration:
                                pass
                                
                            try:
                                next(ask_iter)
                                has_asks = True
                            except StopIteration:
                                pass
                                
                            if has_bids and has_asks:
                                return connector
                        except Exception as e:
                            _logger.warning(f"Error checking order book data for {trading_pair}: {e}")
                            # 继续尝试重新创建连接器
            except Exception as e:
                _logger.warning(f"Connector validation failed for {connector_name}: {e}")
        
        # 创建新的连接器
        exchange_config = EXCHANGES[connector_name]
        
        # 检查API密钥是否提供
        required_params = exchange_config["required_params"]
        missing_params = [key for key, value in required_params.items() if not value]
        if missing_params:
            raise ValueError(f"Missing required API credentials for {connector_name}: {missing_params}")
        
        # 创建客户端配置
        from hummingbot.client.config.client_config_map import ClientConfigMap
        from hummingbot.client.config.config_helpers import ClientConfigAdapter
        
        client_config = ClientConfigMap()
        
        # 配置代理
        proxy = os.environ.get("HTTP_PROXY")
        if proxy:
            _logger.info(f"Using HTTP proxy: {proxy}")
            # Note: Proxy is handled by aiohttp.ClientSession in the connector
        
        client_config_adapter = ClientConfigAdapter(client_config)
        
        # 准备交易所连接参数
        init_params = {
            "client_config_map": client_config_adapter,
            "trading_pairs": [trading_pair],
            "trading_required": True
        }
        
        # 添加交易所特定的必要参数
        init_params.update(required_params)
        
        # 创建连接器实例
        exchange_class = exchange_config["exchange_class"]
        connector = exchange_class(**init_params)
        
        # 等待连接器初始化完成
        try:
            await connector.start_network()
            
            # 检查网络连接状态
            from hummingbot.core.network_iterator import NetworkStatus
            max_network_retries = 3
            network_retry_count = 0
            
            while network_retry_count < max_network_retries:
                try:
                    network_status = await connector.check_network()
                    if network_status == NetworkStatus.CONNECTED:
                        break
                    else:
                        network_retry_count += 1
                        _logger.warning(f"Network not connected for {connector_name} (attempt {network_retry_count}/{max_network_retries}): {network_status}")
                        
                        if network_retry_count >= max_network_retries:
                            raise RuntimeError(f"Network connection failed after {max_network_retries} attempts: {network_status}")
                        else:
                            await asyncio.sleep(1)
                except Exception as e:
                    network_retry_count += 1
                    _logger.warning(f"Network check failed for {connector_name} (attempt {network_retry_count}/{max_network_retries}): {e}")
                    
                    if network_retry_count >= max_network_retries:
                        raise RuntimeError(f"Network connection failed after {max_network_retries} attempts: {e}")
                    else:
                        await asyncio.sleep(1)
            
            _logger.info(f"Connector {connector_name} network initialized successfully")
            
            # 关键修复：手动触发 tick 来启动状态轮询
            # 这是 Hummingbot 中 Clock 机制的核心功能
            import time
            current_time = time.time()
            
            # 触发一次 tick 来启动状态轮询循环
            _logger.info(f"Triggering initial tick to start status polling for {connector_name}")
            connector.tick(current_time)
            await asyncio.sleep(2)  # 等待状态轮询任务启动
            
            # 尝试更新交易规则，但允许失败（使用指数退避）
            max_retries = 3
            retry_count = 0
            trading_rules_updated = False
            
            while retry_count < max_retries:
                try:
                    await connector._update_trading_rules()
                    _logger.info(f"Trading rules updated successfully for {connector_name}")
                    trading_rules_updated = True
                    break
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    # 检查是否是网络连接错误
                    is_network_error = any(err in error_msg.lower() for err in [
                        'contentlengtherror', 'connection', 'timeout', 'network', 'http'
                    ])
                    
                    if is_network_error:
                        _logger.warning(f"Network error updating trading rules for {connector_name} (attempt {retry_count}/{max_retries}): {error_msg}")
                    else:
                        _logger.warning(f"Failed to update trading rules for {connector_name} (attempt {retry_count}/{max_retries}): {e}")
                    
                    if retry_count >= max_retries:
                        _logger.warning(f"Failed to update trading rules for {connector_name} after {max_retries} attempts.")
                        # 如果交易规则更新失败，尝试手动初始化一些基本规则
                        try:
                            _logger.info(f"Attempting to initialize basic trading rules for {connector_name}")
                            # 手动添加一些基本的交易规则
                            from hummingbot.core.data_type.trade_fee import TradeFeeSchema
                            from hummingbot.connector.trading_rule import TradingRule
                            from decimal import Decimal
                            
                            # 为 BTC-USDT 添加基本交易规则
                            basic_rule = TradingRule(
                                trading_pair="BTC-USDT",
                                min_order_size=Decimal("0.00001"),
                                min_price_increment=Decimal("0.01"),
                                min_base_amount_increment=Decimal("0.00001"),
                                min_notional_size=Decimal("10.0")
                            )
                            connector._trading_rules["BTC-USDT"] = basic_rule
                            _logger.info(f"Added basic trading rule for BTC-USDT on {connector_name}")
                            trading_rules_updated = True
                        except Exception as rule_e:
                            _logger.warning(f"Failed to add basic trading rules: {rule_e}")
                    else:
                        # 指数退避：等待时间随重试次数增加
                        wait_time = min(2 ** retry_count, 10)  # 最大等待10秒
                        _logger.debug(f"Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
            
        except Exception as e:
            _logger.error(f"Failed to initialize connector {connector_name}: {e}")
            raise RuntimeError(f"Connector {connector_name} failed to initialize: {e}")
        
        # 等待连接器完全就绪（包括账户余额）
        if wait_for_orderbook or connector.is_trading_required:
            max_wait_time = 30  # 等待时间30秒
            wait_time = 0
            last_error = None
            last_tick_time = time.time()
            
            while wait_time < max_wait_time:
                try:
                    # 每10秒触发一次 tick 确保状态轮询继续工作（降低频率避免网络问题）
                    current_time = time.time()
                    if current_time - last_tick_time >= 10:
                        connector.tick(current_time)
                        last_tick_time = current_time
                        _logger.debug(f"Triggered periodic tick for {connector_name}")
                    
                    # 检查连接器状态
                    status_dict = connector.status_dict
                    _logger.debug(f"Connector {connector_name} status: {status_dict}")
                    
                    if not connector.ready:
                        # 详细记录哪些组件未就绪
                        not_ready_components = [k for k, v in status_dict.items() if not v]
                        last_error = f"Connector components not ready after {wait_time}s: {not_ready_components}"
                        _logger.warning(f"Connector {connector_name} not ready: {status_dict}")
                        await asyncio.sleep(2)
                        wait_time += 2
                        continue
                    
                    # 检查订单簿跟踪器状态（如果需要）
                    if wait_for_orderbook:
                        if not hasattr(connector, 'order_book_tracker') or not connector.order_book_tracker.ready:
                            last_error = f"Order book tracker not ready after {wait_time}s"
                            _logger.warning(f"Order book tracker not ready for {connector_name}")
                            await asyncio.sleep(2)
                            wait_time += 2
                            continue
                        
                        # 检查交易对是否存在
                        if trading_pair not in connector.order_book_tracker.order_books:
                            last_error = f"Trading pair {trading_pair} not found in order books after {wait_time}s"
                            available_pairs = list(connector.order_book_tracker.order_books.keys())
                            _logger.warning(f"Trading pair {trading_pair} not found. Available pairs: {available_pairs[:5]}...")
                            await asyncio.sleep(2)
                            wait_time += 2
                            continue
                        
                        # 检查订单簿数据
                        order_book = connector.order_book_tracker.order_books[trading_pair]
                        if not order_book:
                            last_error = f"Order book is None for {trading_pair} after {wait_time}s"
                            await asyncio.sleep(2)
                            wait_time += 2
                            continue
                        
                        # 正确处理 Cython 生成器 - 不能直接使用 len()
                        try:
                            # 使用迭代器方式检查是否有数据，避免转换为列表的性能问题
                            bid_entries_iter = order_book.bid_entries()
                            ask_entries_iter = order_book.ask_entries()
                            
                            # 检查是否至少有一个条目
                            try:
                                next(bid_entries_iter)
                                bid_count = 1  # 至少有一个bid
                            except StopIteration:
                                bid_count = 0
                                
                            try:
                                next(ask_entries_iter)
                                ask_count = 1  # 至少有一个ask
                            except StopIteration:
                                ask_count = 0
                                
                        except Exception as entries_error:
                            _logger.warning(f"Error getting order book entries: {entries_error}")
                            # 尝试其他方法检查订单簿
                            try:
                                # 检查订单簿是否有数据的替代方法
                                if hasattr(order_book, 'get_price') and order_book.get_price(True) is not None and order_book.get_price(False) is not None:
                                    bid_count = 1  # 至少有数据
                                    ask_count = 1
                                else:
                                    bid_count = 0
                                    ask_count = 0
                            except Exception:
                                bid_count = 0
                                ask_count = 0
                        
                        if bid_count == 0 or ask_count == 0:
                            last_error = f"Order book incomplete: {bid_count} bids, {ask_count} asks after {wait_time}s"
                            await asyncio.sleep(2)
                            wait_time += 2
                            continue
                        
                        _logger.info(f"Order book initialized successfully for {trading_pair} on {connector_name}: {bid_count} bids, {ask_count} asks")
                    
                    # 检查账户余额（对于交易功能必需）
                    if connector.is_trading_required:
                        if len(connector._account_balances) == 0:
                            last_error = f"Account balances not loaded after {wait_time}s"
                            _logger.warning(f"Account balances not loaded for {connector_name}")
                            await asyncio.sleep(2)
                            wait_time += 2
                            continue
                        else:
                            _logger.info(f"Account balances loaded for {connector_name}: {len(connector._account_balances)} assets")
                    
                    # 所有检查通过，跳出循环
                    _logger.info(f"Connector {connector_name} fully initialized and ready")
                    break
                    
                except Exception as e:
                    last_error = f"Error during initialization: {str(e)}"
                    _logger.warning(f"Error during order book initialization: {e}")
                    await asyncio.sleep(2)
                    wait_time += 2
            
            if wait_time >= max_wait_time:
                error_msg = f"Failed to initialize data for {trading_pair} on {connector_name}"
                if last_error:
                    error_msg += f" - {last_error}"
                
                # 对于交易功能，account_balance必须准备好
                if connector.is_trading_required:
                    status_dict = connector.status_dict
                    if not status_dict.get('account_balance', False):
                        error_msg += " - Account balance not loaded (required for trading)"
                        _logger.error(f"Trading initialization timeout: {error_msg}")
                        raise RuntimeError(error_msg)
                
                # 对于非交易功能或仅订单簿问题，允许继续
                _logger.warning(f"Order book initialization timeout: {error_msg}")
                _logger.info(f"Connector {connector_name} initialized with limited functionality")
        else:
            # 不等待订单簿数据，但仍需检查账户余额（如果需要交易功能）
            if connector.is_trading_required:
                _logger.info(f"Quick check account balance for {connector_name}...")
                
                # 触发 tick 并等待账户余额加载（快速检查）
                # 使用更合理的检查频率，避免过多网络请求
                for i in range(3):  # 最多检查3次，总共15秒
                    if i == 0:
                        # 第一次立即检查
                        connector.tick(time.time())
                        await asyncio.sleep(3)
                    else:
                        # 后续检查间隔更长
                        await asyncio.sleep(5)
                        connector.tick(time.time())
                        await asyncio.sleep(1)
                    
                    if len(connector._account_balances) > 0:
                        _logger.info(f"Account balances loaded for {connector_name}: {len(connector._account_balances)} assets")
                        break
                    else:
                        _logger.debug(f"Attempt {i+1}/3 - account balances not yet loaded for {connector_name}")
                else:
                    # 对于交易功能，账户余额是必需的
                    error_msg = f"Account balances not loaded for {connector_name} after 3 attempts (15s). Trading functionality requires account balance information."
                    _logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            # 检查连接器基本状态
            status_dict = connector.status_dict
            _logger.info(f"Connector {connector_name} status: {status_dict}")
            
            # 检查关键组件是否就绪
            if not connector.ready:
                _logger.warning(f"Connector {connector_name} not fully ready, but continuing with available components")
                # 检查哪些组件未就绪
                for component, status in status_dict.items():
                    if not status:
                        _logger.warning(f"Component {component} not ready for {connector_name}")
            
            _logger.info(f"Connector {connector_name} initialized without waiting for complete order book data")
        
        # 缓存连接器
        _connectors[connector_name] = connector
        
        return connector


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

                proxy = os.environ.get("HTTP_PROXY")
                # 创建带代理的 ClientSession
                connector = TCPConnector(ssl=False)
                _session = ClientSession(
                    connector=connector,
                    timeout=ClientTimeout(total=30),
                    trust_env=True,
                    proxy=proxy
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
    global _order_book_tracker, _initialized, _throttler, _session, _connectors

    try:
        if _order_book_tracker is not None:
            _order_book_tracker.stop()
        if _session is not None:
            await _session.close()
        
        # 清理连接器缓存
        for connector_name, connector in _connectors.items():
            try:
                await connector.stop_network()
            except Exception as e:
                _logger.warning(f"Error stopping connector {connector_name}: {e}")
        
        _connectors.clear()
        _connector_locks.clear()
        _initialized = False
        _logger.info("Successfully cleaned up resources")
    except Exception as e:
        _logger.error(f"Error during cleanup: {e}")
        raise


def clear_connector_cache(connector_name: str = None):
    """
    清理连接器缓存
    
    参数:
        connector_name (str): 指定要清理的交易所，如果为None则清理所有
    """
    global _connectors, _connector_locks
    
    if connector_name is None:
        # 清理所有连接器
        _connectors.clear()
        _connector_locks.clear()
        _logger.info("Cleared all connector cache")
    elif connector_name in _connectors:
        # 清理指定连接器
        del _connectors[connector_name]
        if connector_name in _connector_locks:
            del _connector_locks[connector_name]
        _logger.info(f"Cleared connector cache for {connector_name}")
    else:
        _logger.warning(f"No connector cache found for {connector_name}")


async def cleanup_inactive_connectors():
    """
    清理不活跃的连接器
    """
    global _connectors
    
    inactive_connectors = []
    for connector_name, connector in _connectors.items():
        try:
            # 检查连接器是否仍然活跃
            status_dict = connector.status_dict
            if not any(status_dict.values()):  # 如果所有状态都是 False
                inactive_connectors.append(connector_name)
        except Exception as e:
            _logger.warning(f"Error checking connector {connector_name}: {e}")
            inactive_connectors.append(connector_name)
    
    # 清理不活跃的连接器
    for connector_name in inactive_connectors:
        try:
            connector = _connectors.get(connector_name)
            if connector:
                await connector.stop_network()
            clear_connector_cache(connector_name)
            _logger.info(f"Cleaned up inactive connector: {connector_name}")
        except Exception as e:
            _logger.warning(f"Error cleaning up connector {connector_name}: {e}")
    
    return len(inactive_connectors)


async def force_refresh_connector_data(connector_name: str, trading_pair: str):
    """
    强制刷新连接器数据，确保获取最新数据
    
    参数:
        connector_name (str): 交易所名称
        trading_pair (str): 交易对
        
    返回:
        连接器实例
    """
    # 清除指定连接器的缓存
    clear_connector_cache(connector_name)
    
    # 重新创建连接器，获取最新数据
    return await get_or_create_connector(connector_name, trading_pair)


async def test_connector_connection(connector_name: str):
    """
    测试连接器连接状态
    
    参数:
        connector_name (str): 交易所名称
        
    返回:
        dict: 连接测试结果
    """
    try:
        if connector_name not in EXCHANGES:
            return {
                "success": False,
                "error": f"Unsupported exchange: {connector_name}"
            }
        
        exchange_config = EXCHANGES[connector_name]
        
        # 检查API密钥
        required_params = exchange_config["required_params"]
        missing_params = [key for key, value in required_params.items() if not value]
        if missing_params:
            return {
                "success": False,
                "error": f"Missing API credentials: {missing_params}",
                "suggestion": "Please check your .env file and ensure API keys are set correctly"
            }
        
        # 创建客户端配置
        from hummingbot.client.config.client_config_map import ClientConfigMap
        from hummingbot.client.config.config_helpers import ClientConfigAdapter
        
        client_config = ClientConfigMap()
        
        # 配置代理
        proxy = os.environ.get("HTTP_PROXY")
        if proxy:
            _logger.info(f"Using HTTP proxy: {proxy}")
            # Note: Proxy is handled by aiohttp.ClientSession in the connector
        
        client_config_adapter = ClientConfigAdapter(client_config)
        
        # 准备交易所连接参数
        init_params = {
            "client_config_map": client_config_adapter,
            "trading_pairs": ["BTC-USDT"],  # 使用简单的测试交易对
            "trading_required": False  # 不需要交易功能，只测试连接
        }
        
        # 添加交易所特定的必要参数
        init_params.update(required_params)
        
        # 创建连接器实例
        exchange_class = exchange_config["exchange_class"]
        connector = exchange_class(**init_params)
        
        # 测试网络连接
        try:
            await connector.start_network()
            network_status = await connector.check_network()
            
            # 检查网络状态枚举值
            from hummingbot.core.network_iterator import NetworkStatus
            if network_status == NetworkStatus.CONNECTED:
                return {
                    "success": True,
                    "message": f"Successfully connected to {connector_name}",
                    "network_status": str(network_status),
                    "connector_ready": connector.ready
                }
            else:
                return {
                    "success": False,
                    "error": f"Network connection failed: {network_status}",
                    "network_status": str(network_status)
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Connection test failed: {str(e)}",
                "suggestion": "Check your internet connection and API credentials"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to test connection: {str(e)}"
        }


def check_connector_data_freshness(connector_name: str, trading_pair: str):
    """
    检查连接器数据的新鲜度
    
    参数:
        connector_name (str): 交易所名称
        trading_pair (str): 交易对
        
    返回:
        dict: 数据新鲜度信息
    """
    if connector_name not in _connectors:
        return {
            "fresh": False,
            "reason": "Connector not cached",
            "connector_name": connector_name
        }
    
    connector = _connectors[connector_name]
    
    try:
        # 检查连接器状态
        if not hasattr(connector, 'ready') or not connector.ready:
            return {
                "fresh": False,
                "reason": "Connector not ready",
                "connector_name": connector_name
            }
        
        # 检查订单簿跟踪器状态
        if not hasattr(connector, 'order_book_tracker') or not connector.order_book_tracker.ready:
            return {
                "fresh": False,
                "reason": "Order book tracker not ready",
                "connector_name": connector_name
            }
        
        # 检查交易对是否存在
        if trading_pair not in connector.order_book_tracker.order_books:
            return {
                "fresh": False,
                "reason": f"Trading pair {trading_pair} not found",
                "connector_name": connector_name
            }
        
        # 检查订单簿数据
        order_book = connector.order_book_tracker.order_books[trading_pair]
        if not order_book:
            return {
                "fresh": False,
                "reason": "Order book is None",
                "connector_name": connector_name,
                "trading_pair": trading_pair
            }
            
        try:
            # 使用迭代器方式检查是否有数据，避免len()的问题
            bid_iter = order_book.bid_entries()
            ask_iter = order_book.ask_entries()
            
            # 检查是否至少有一个条目
            has_bids = False
            has_asks = False
            bid_count = 0
            ask_count = 0
            
            try:
                next(bid_iter)
                has_bids = True
                bid_count = 1  # 至少有一个
            except StopIteration:
                pass
                
            try:
                next(ask_iter)
                has_asks = True
                ask_count = 1  # 至少有一个
            except StopIteration:
                pass
                
            if not has_bids or not has_asks:
                return {
                    "fresh": False,
                    "reason": "Order book data incomplete",
                    "connector_name": connector_name,
                    "trading_pair": trading_pair
                }
        except Exception as e:
            return {
                "fresh": False,
                "reason": f"Error checking order book data: {str(e)}",
                "connector_name": connector_name,
                "trading_pair": trading_pair
            }
        
        return {
            "fresh": True,
            "reason": "Data is fresh and complete",
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "bid_count": bid_count,
            "ask_count": ask_count
        }
        
    except Exception as e:
        return {
            "fresh": False,
            "reason": f"Error checking data freshness: {str(e)}",
            "connector_name": connector_name,
            "trading_pair": trading_pair
        }


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

        # 获取买卖单 - 正确处理 Cython 生成器
        try:
            # 将生成器转换为列表，然后处理
            bid_entries = list(order_book.bid_entries())
            ask_entries = list(order_book.ask_entries())
            
            bids = [(str(price), str(amount)) for price, amount, _ in bid_entries]
            asks = [(str(price), str(amount)) for price, amount, _ in ask_entries]
            
        except Exception as e:
            _logger.warning(f"Error getting order book entries for {token}: {e}")
            # 尝试使用替代方法获取价格数据
            try:
                bid_price = order_book.get_price(True)  # 买入价格
                ask_price = order_book.get_price(False)  # 卖出价格
                if bid_price is not None and ask_price is not None:
                    # 创建模拟的订单簿数据
                    bids = [(str(bid_price), "1.0")]  # 模拟数据
                    asks = [(str(ask_price), "1.0")]
                else:
                    return None
            except Exception:
                _logger.error(f"Failed to get order book data for {token}")
                return None
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
    elif chain == "SOL":
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
        # average_price = average_price * 1.001  # 0.1% 交易所手续费
        price_impact = analysis['price_impact']
        fluxlayer_price = average_price
        
        # 根据买入/卖出操作调整价格
        # if is_buy:
        #     # 买入时，fluxlayer作为卖方，提高价格
        #     if price_impact * 10000 > 10:  # fluxlayer 抽成
        #         fluxlayer_price = average_price * 1.01
        #     else:
        #         fluxlayer_price = average_price * 1.004
        # else:
        #     # 卖出时，fluxlayer作为买方，降低价格
        #     if price_impact * 10000 > 10:  # fluxlayer 抽成
        #         fluxlayer_price = average_price / 1.01
        #     else:
        #         fluxlayer_price = average_price / 1.004
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
    # 注意：不在这里调用 cleanup()，因为连接器需要被缓存重用
    # cleanup() 应该在程序结束时或明确需要清理时调用

getcontext().prec = 28

# 币种精度表
TOKEN_PRECISION = {
    "BTC": 8,
    "ETH": 18,
    "USDT": 6,
    "SOL": 9,
}

def truncate_amount(amount: str, token: str) -> str:
    precision = TOKEN_PRECISION.get(token.upper(), 8)  # 默认8位
    quantize_str = '1.' + '0' * precision  # 如 '1.00000000'
    return str(Decimal(amount).quantize(Decimal(quantize_str), rounding=ROUND_DOWN))

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
        best_result["target_amount"] = truncate_amount(best_result["target_amount"], target_token)

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
                        AND supported_asset ? $1
                        AND supported_asset ? $2
                        AND supported_asset -> $1 ? $3
                        AND supported_asset -> $2 ? $4
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
                timeout = ClientTimeout(total=30)  # 5秒超时
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
    try:
        import asyncpg
        return await asyncpg.connect(
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", "postgres"),
            database=os.getenv("PG_DATABASE", "postgres"),
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", 5432)),
        )
    except ImportError:
        raise ImportError("asyncpg module is required for database operations. Install it with: pip install asyncpg")


async def place_order_on_exchange(
    connector_name: str,
    trading_pair: str,
    amount: float,
    is_buy: bool,
    order_type: str = "MARKET",
    price: float = None,
    **kwargs
):
    """
    在指定的CEX连接器上执行实际下单操作
    
    参数:
        connector_name (str): 交易所名称，如 "binance"、"hyperliquid" 等
        trading_pair (str): 交易对，如 "BTC-USDT"
        amount (float): 下单数量（基础资产数量）
            - 买入时：要买入的基础资产数量（如 BTC 0.001）
            - 卖出时：要卖出的基础资产数量（如 BTC 0.001）
            - 注意：这是基础资产的数量，不是USDT金额
        is_buy (bool): True=买入，False=卖出  
        order_type (str): 订单类型，"MARKET"/"LIMIT"/"LIMIT_MAKER"
        price (float): 订单价格
            - MARKET: 应为 None，由交易所决定执行价格
            - LIMIT/LIMIT_MAKER: 必须提供有效价格（> 0）
        **kwargs: 其他参数
        
    返回:
        dict: 包含订单信息的结果
        {
            "success": bool,
            "order_id": str,
            "exchange": str,
            "trading_pair": str,
            "amount": float,
            "price": float,
            "side": str,
            "order_type": str,
            "error": str (如果失败）
        }
        
    示例:
        # 买入 0.001 BTC（不是 0.001 USDT）
        place_order_on_exchange("binance", "BTC-USDT", 0.001, True, "MARKET")
        
        # 卖出 0.1 ETH（不是 0.1 USDT）
        place_order_on_exchange("binance", "ETH-USDT", 0.1, False, "LIMIT", 2500.0)
    """
    try:
        # 参数验证
        if connector_name not in EXCHANGES:
            return {
                "success": False,
                "error": f"Unsupported exchange: {connector_name}"
            }
            
        if amount <= 0:
            return {
                "success": False,
                "error": "Amount must be greater than 0"
            }
            
        from hummingbot.core.data_type.common import OrderType
        from hummingbot.client.config.client_config_map import ClientConfigMap
        from hummingbot.client.config.config_helpers import ClientConfigAdapter
        from decimal import Decimal
        
        # 订单类型映射
        order_type_map = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "LIMIT_MAKER": OrderType.LIMIT_MAKER
        }
        
        if order_type not in order_type_map:
            return {
                "success": False,
                "error": f"Unsupported order type: {order_type}. Use MARKET, LIMIT, or LIMIT_MAKER"
            }
            
        exchange_config = EXCHANGES[connector_name]
        
        # 创建客户端配置
        client_config = ClientConfigMap()
        client_config_adapter = ClientConfigAdapter(client_config)
        
        # 准备交易所连接参数
        init_params = {
            "client_config_map": client_config_adapter,
            "trading_pairs": [trading_pair],
            "trading_required": True  # 需要交易功能
        }
        
        # 检查API密钥是否提供
        required_params = exchange_config["required_params"]
        missing_params = [key for key, value in required_params.items() if not value]
        if missing_params:
            return {
                "success": False,
                "error": f"Missing required API credentials for {connector_name}: {missing_params}"
            }
        
        # 添加交易所特定的必要参数（API密钥等）
        init_params.update(required_params)
        
        # 获取或创建连接器实例（使用缓存）
        # 对于市价单，我们需要等待订单簿数据，因为需要获取当前价格
        # 对于限价单，可以跳过订单簿等待
        wait_for_orderbook = (order_type == "MARKET")
        try:
            connector = await get_or_create_connector(connector_name, trading_pair, wait_for_orderbook=wait_for_orderbook)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize connector for {connector_name}: {str(e)}",
                "exchange": connector_name,
                "trading_pair": trading_pair,
                "amount": amount,
                "side": "BUY" if is_buy else "SELL"
            }
        
        # 价格验证和处理
        if order_type == "MARKET":
            # 市价单不需要预先设置价格，让交易所根据当前市场价格执行
            # Hummingbot 原始逻辑：市价单使用 price=None 或 s_decimal_NaN
            price = None  # 让交易所决定执行价格
        elif order_type in ["LIMIT", "LIMIT_MAKER"]:
            # 限价单必须提供价格
            if price is None:
                return {
                    "success": False,
                    "error": f"Price is required for {order_type} orders"
                }
            if price <= 0:
                return {
                    "success": False,
                    "error": f"Price must be greater than 0 for {order_type} orders"
                }
        
        # 最终检查连接器状态（特别是账户余额）
        status_dict = connector.status_dict
        if not status_dict.get('account_balance', False):
            return {
                "success": False,
                "error": f"Account balance not loaded for {connector_name}. Cannot place order without account information.",
                "exchange": connector_name,
                "trading_pair": trading_pair,
                "amount": amount,
                "side": "BUY" if is_buy else "SELL",
                "connector_status": status_dict
            }
        
        if not connector.ready:
            not_ready_components = [k for k, v in status_dict.items() if not v]
            return {
                "success": False,
                "error": f"Connector {connector_name} not ready. Missing components: {not_ready_components}",
                "exchange": connector_name,
                "trading_pair": trading_pair,
                "amount": amount,
                "side": "BUY" if is_buy else "SELL",
                "connector_status": status_dict
            }
        
        _logger.info(f"Connector {connector_name} ready for trading. Status: {status_dict}")
        
        # 执行下单
        # 根据 Hummingbot 原始逻辑处理价格
        if order_type == "MARKET":
            # 市价单使用 None，让 Hummingbot 内部处理
            order_price = None
        else:
            # 限价单使用提供的价格
            order_price = Decimal(str(price)) if price is not None else None
            
        try:
            # 执行下单
            if is_buy:
                order_id = connector.buy(
                    trading_pair=trading_pair,
                    amount=Decimal(str(amount)),
                    order_type=order_type_map[order_type],
                    price=order_price,
                    **kwargs
                )
            else:
                order_id = connector.sell(
                    trading_pair=trading_pair,
                    amount=Decimal(str(amount)),
                    order_type=order_type_map[order_type],
                    price=order_price,
                    **kwargs
                )
            
            _logger.info(f"Order submitted to {connector_name} with local ID: {order_id}")
            
            # 等待订单状态更新，并检查是否真的提交成功
            await asyncio.sleep(2)
            
            # 检查订单是否真的存在于连接器的订单跟踪器中
            order_tracker = connector._order_tracker
            if hasattr(order_tracker, 'all_orders') and order_id in order_tracker.all_orders:
                order = order_tracker.all_orders[order_id]
                _logger.info(f"Order found in tracker: {order_id}, state: {order.current_state}")
                
                # 检查订单状态，如果是失败状态，返回错误
                if hasattr(order, 'current_state'):
                    order_state = str(order.current_state)
                    if 'FAILED' in order_state.upper() or 'CANCELLED' in order_state.upper():
                        # 获取详细的失败原因
                        failure_reason = "Unknown reason"
                        
                        # 尝试从订单对象获取错误信息
                        if hasattr(order, 'last_state') and order.last_state:
                            failure_reason = f"Last state: {order.last_state}"
                        
                        # 检查是否有交易所错误信息
                        if hasattr(order, 'exchange_order_id') and not order.exchange_order_id:
                            failure_reason += " - Order was not submitted to exchange (likely API key or permission issue)"
                        
                        # 检查连接器的最近错误
                        if hasattr(connector, '_last_order_update_timestamp'):
                            try:
                                # 尝试获取连接器的错误日志
                                recent_errors = []
                                if hasattr(connector, '_order_not_found_records'):
                                    recent_errors.extend(list(connector._order_not_found_records.values())[-3:])
                                
                                if recent_errors:
                                    failure_reason += f" - Recent errors: {recent_errors}"
                            except Exception:
                                pass
                                
                        # 检查网络状态
                        try:
                            network_status = await connector.check_network()
                            if str(network_status) != "NetworkStatus.CONNECTED":
                                failure_reason += f" - Network status: {network_status}"
                        except Exception as e:
                            failure_reason += f" - Network check failed: {str(e)}"
                        
                        return {
                            "success": False,
                            "error": f"Order failed with state: {order_state}",
                            "failure_reason": failure_reason,
                            "exchange": connector_name,
                            "trading_pair": trading_pair,
                            "amount": amount,
                            "side": "BUY" if is_buy else "SELL",
                            "order_id": order_id,
                            "exchange_order_id": getattr(order, 'exchange_order_id', None),
                            "order_details": {
                                "creation_timestamp": getattr(order, 'creation_timestamp', None),
                                "last_update_timestamp": getattr(order, 'last_update_timestamp', None),
                                "executed_amount_base": float(getattr(order, 'executed_amount_base', 0)),
                                "executed_amount_quote": float(getattr(order, 'executed_amount_quote', 0)),
                            }
                        }
            else:
                _logger.warning(f"Order {order_id} not found in order tracker for {connector_name}")
                # 订单不在跟踪器中，可能提交失败了
                return {
                    "success": False,
                    "error": f"Order not found in tracker after submission. Possible network or API key issue.",
                    "exchange": connector_name,
                    "trading_pair": trading_pair,
                    "amount": amount,
                    "side": "BUY" if is_buy else "SELL",
                    "order_id": order_id
                }
            
            return {
                "success": True,
                "order_id": order_id,
                "exchange": connector_name,
                "trading_pair": trading_pair,
                "amount": amount,
                "price": price,
                "side": "BUY" if is_buy else "SELL",
                "order_type": order_type,
                "timestamp": time.time()
            }
            
        except Exception as order_error:
            _logger.error(f"Error during order execution on {connector_name}: {order_error}")
            return {
                "success": False,
                "error": f"Order execution failed: {str(order_error)}",
                "exchange": connector_name,
                "trading_pair": trading_pair,
                "amount": amount,
                "side": "BUY" if is_buy else "SELL"
            }
        
    except Exception as e:
        _logger.error(f"Error placing order on {connector_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "exchange": connector_name,
            "trading_pair": trading_pair,
            "amount": amount,
            "side": "BUY" if is_buy else "SELL"
        }


async def get_order_status(connector_name: str, order_id: str, trading_pair: str):
    """
    查询订单状态
    
    参数:
        connector_name (str): 交易所名称
        order_id (str): 订单ID
        trading_pair (str): 交易对
        
    返回:
        dict: 订单状态信息
    """
    try:
        if connector_name not in EXCHANGES:
            return {
                "success": False,
                "error": f"Unsupported exchange: {connector_name}"
            }
            
        exchange_config = EXCHANGES[connector_name]
        
        # 创建客户端配置
        from hummingbot.client.config.client_config_map import ClientConfigMap
        from hummingbot.client.config.config_helpers import ClientConfigAdapter
        
        client_config = ClientConfigMap()
        client_config_adapter = ClientConfigAdapter(client_config)
        
        # 准备交易所连接参数
        init_params = {
            "client_config_map": client_config_adapter,
            "trading_pairs": [trading_pair],
            "trading_required": True
        }
        
        # 添加交易所特定的必要参数
        init_params.update(exchange_config["required_params"])
        
        # 获取或创建连接器实例（使用缓存）
        connector = await get_or_create_connector(connector_name, trading_pair)
        
        # 获取订单状态
        # 注意：不同交易所的订单状态查询方法可能不同
        # 这里提供一个通用的框架，具体实现需要根据各交易所的API调整
        order_tracker = connector._order_tracker
        order = order_tracker.all_orders.get(order_id)
        
        if order:
            order_state = str(order.current_state)
            result = {
                "success": True,
                "order_id": order_id,
                "exchange": connector_name,
                "trading_pair": trading_pair,
                "status": order_state,
                "filled_amount": float(order.executed_amount_base),
                "remaining_amount": float(order.amount - order.executed_amount_base),
                "average_price": float(order.average_executed_price) if order.average_executed_price else None,
                "created_timestamp": order.creation_timestamp,
                "last_update_timestamp": order.last_update_timestamp,
                "exchange_order_id": getattr(order, 'exchange_order_id', None)
            }
            
            # 如果订单失败，添加详细的失败信息
            if 'FAILED' in order_state.upper() or 'CANCELLED' in order_state.upper():
                failure_reason = "Unknown reason"
                
                # 尝试从订单对象获取错误信息
                if hasattr(order, 'last_state') and order.last_state:
                    failure_reason = f"Last state: {order.last_state}"
                
                # 检查是否有交易所错误信息
                if hasattr(order, 'exchange_order_id') and not order.exchange_order_id:
                    failure_reason += " - Order was not submitted to exchange (likely API key or permission issue)"
                    
                # 检查订单是否有特定的失败原因
                if hasattr(order, 'fail_reason') and order.fail_reason:
                    failure_reason += f" - Fail reason: {order.fail_reason}"
                
                result["failure_reason"] = failure_reason
                result["is_failure"] = True
            else:
                result["is_failure"] = False
                
            return result
        else:
            return {
                "success": False,
                "error": f"Order {order_id} not found",
                "order_id": order_id,
                "exchange": connector_name
            }
            
    except Exception as e:
        _logger.error(f"Error getting order status from {connector_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "order_id": order_id,
            "exchange": connector_name
        }


async def cancel_order_on_exchange(connector_name: str, order_id: str, trading_pair: str):
    """
    取消订单
    
    参数:
        connector_name (str): 交易所名称
        order_id (str): 订单ID
        trading_pair (str): 交易对
        
    返回:
        dict: 取消结果
    """
    try:
        if connector_name not in EXCHANGES:
            return {
                "success": False,
                "error": f"Unsupported exchange: {connector_name}"
            }
            
        exchange_config = EXCHANGES[connector_name]
        
        # 创建客户端配置
        from hummingbot.client.config.client_config_map import ClientConfigMap
        from hummingbot.client.config.config_helpers import ClientConfigAdapter
        
        client_config = ClientConfigMap()
        
        # 配置代理
        proxy = os.environ.get("HTTP_PROXY")
        if proxy:
            _logger.info(f"Using HTTP proxy: {proxy}")
            # Note: Proxy is handled by aiohttp.ClientSession in the connector
        
        client_config_adapter = ClientConfigAdapter(client_config)
        
        # 准备交易所连接参数
        init_params = {
            "client_config_map": client_config_adapter,
            "trading_pairs": [trading_pair],
            "trading_required": True
        }
        
        # 添加交易所特定的必要参数
        init_params.update(exchange_config["required_params"])
        
        # 获取或创建连接器实例（使用缓存）
        connector = await get_or_create_connector(connector_name, trading_pair)
        
        # 取消订单
        connector.cancel(trading_pair, order_id)
        
        # 等待一小段时间让订单状态更新
        await asyncio.sleep(1)
        
        return {
            "success": True,
            "order_id": order_id,
            "exchange": connector_name,
            "trading_pair": trading_pair,
            "timestamp": time.time()
        }
        
    except Exception as e:
        _logger.error(f"Error canceling order on {connector_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "order_id": order_id,
            "exchange": connector_name
        }


def get_supported_exchanges():
    """
    获取支持的交易所列表
    
    返回:
        list: 支持的交易所名称列表
    """
    return list(EXCHANGES.keys())


def get_exchange_info(connector_name: str):
    """
    获取交易所信息
    
    参数:
        connector_name (str): 交易所名称
        
    返回:
        dict: 交易所配置信息
    """
    if connector_name not in EXCHANGES:
        return {
            "error": f"Unsupported exchange: {connector_name}"
        }
    
    config = EXCHANGES[connector_name]
    return {
        "name": connector_name,
        "exchange_class": config["exchange_class"].__name__,
        "data_source_class": config["data_source_class"].__name__,
        "required_params": list(config["required_params"].keys())
    }


async def get_connector_diagnostic_info(connector_name: str):
    """
    获取连接器的详细诊断信息
    
    参数:
        connector_name (str): 交易所名称
        
    返回:
        dict: 详细的诊断信息
    """
    try:
        if connector_name not in EXCHANGES:
            return {
                "error": f"Unsupported exchange: {connector_name}"
            }
        
        diagnostic_info = {
            "connector_name": connector_name,
            "cached": connector_name in _connectors,
            "exchange_config": get_exchange_info(connector_name)
        }
        
        # 如果连接器已缓存，获取详细状态
        if connector_name in _connectors:
            connector = _connectors[connector_name]
            
            # 基本状态信息
            diagnostic_info.update({
                "ready": connector.ready if hasattr(connector, 'ready') else False,
                "status_dict": connector.status_dict if hasattr(connector, 'status_dict') else {},
                "trading_required": connector.is_trading_required if hasattr(connector, 'is_trading_required') else False,
            })
            
            # 网络状态
            try:
                network_status = await connector.check_network()
                diagnostic_info["network_status"] = str(network_status)
            except Exception as e:
                diagnostic_info["network_status"] = f"Error: {str(e)}"
            
            # 账户余额信息
            if hasattr(connector, '_account_balances'):
                diagnostic_info["account_balances_count"] = len(connector._account_balances)
                diagnostic_info["account_balances"] = dict(list(connector._account_balances.items())[:5])  # 只显示前5个
            
            # 订单跟踪器信息
            if hasattr(connector, '_order_tracker'):
                order_tracker = connector._order_tracker
                diagnostic_info["active_orders"] = len(order_tracker.all_orders) if hasattr(order_tracker, 'all_orders') else 0
                
                # 获取最近的订单信息
                if hasattr(order_tracker, 'all_orders') and order_tracker.all_orders:
                    recent_orders = list(order_tracker.all_orders.items())[-3:]  # 最近3个订单
                    diagnostic_info["recent_orders"] = [
                        {
                            "order_id": order_id,
                            "state": str(order.current_state),
                            "exchange_order_id": getattr(order, 'exchange_order_id', None),
                            "amount": float(order.amount) if hasattr(order, 'amount') else None,
                            "executed_amount": float(order.executed_amount_base) if hasattr(order, 'executed_amount_base') else None,
                        }
                        for order_id, order in recent_orders
                    ]
            
            # 订单簿信息
            if hasattr(connector, 'order_book_tracker') and connector.order_book_tracker:
                tracker = connector.order_book_tracker
                diagnostic_info["order_book_tracker_ready"] = tracker.ready if hasattr(tracker, 'ready') else False
                diagnostic_info["order_books_count"] = len(tracker.order_books) if hasattr(tracker, 'order_books') else 0
                if hasattr(tracker, 'order_books'):
                    diagnostic_info["available_trading_pairs"] = list(tracker.order_books.keys())[:10]  # 前10个交易对
        
        return diagnostic_info
        
    except Exception as e:
        return {
            "error": f"Error getting diagnostic info: {str(e)}",
            "connector_name": connector_name
        }


# 使用示例
async def example_usage():
    # 示例1：在 Binance 上买入 0.001 BTC（市价单）
    # 注意：amount 是 BTC 数量，不是 USDT 金额
    # 市价单不需要设置 price，由交易所决定执行价格
    result = await place_order_on_exchange(
        connector_name="binance",
        trading_pair="BTC-USDT",
        amount=0.0000546,  # 买入 0.001 BTC (满足最小订单量要求)
        is_buy=True,
        order_type="MARKET"
    )
    print("Market Buy Order Result:", result)
    
    # 示例2：在 Binance 上卖出 0.001 BTC（限价单）
    # 限价单必须提供价格
    # result = await place_order_on_exchange(
    #     connector_name="binance",
    #     trading_pair="BTC-USDT",
    #     amount=0.001,  # 卖出 0.001 BTC
    #     is_buy=False,
    #     order_type="LIMIT",
    #     price=50000.0  # 限价单必须提供价格
    # )
    # print("Limit Sell Order Result:", result)
    
    # 示例2：在 Hyperliquid 上卖出 0.1 ETH（限价单）
    # result = await place_order_on_exchange(
    #     connector_name="hyperliquid",
    #     trading_pair="ETH-USDT",
    #     amount=0.1,
    #     is_buy=False,
    #     order_type="LIMIT",
    #     price=2500.0
    # )
    # print("Limit Sell Order Result:", result)
    
    if result.get("success"):
        order_id = result["order_id"]
        # 查询订单状态
        status = await get_order_status("binance", order_id, "BTC-USDT")
        print("Order Status:", status)
    else:
        print("Order Failed!")
        print("Error:", result.get("error"))
        if "failure_reason" in result:
            print("Failure Reason:", result["failure_reason"])
        if "order_details" in result:
            print("Order Details:", result["order_details"])
        
        # 获取详细的连接器诊断信息来帮助调试
        print("\n--- Diagnostic Information ---")
        diagnostic = await get_connector_diagnostic_info("binance")
        print("Connector Diagnostic:", diagnostic)
        
        # 如果需要，可以取消订单
        # cancel_result = await cancel_order_on_exchange("binance", order_id, "BTC-USDT")
        # print("Cancel Result:", cancel_result)

if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())
