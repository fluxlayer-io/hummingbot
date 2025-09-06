import asyncio
import json
import os
import sys
import time
from datetime import datetime
from decimal import ROUND_DOWN, Decimal, getcontext

import aiohttp
import asyncpg
import requests
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from hummingbot.client.config.client_config_map import AnonymizedMetricsEnabledMode, ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.client.hummingbot_application import HummingbotApplication
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.data_type.order_book_tracker import OrderBookTracker
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.fluxlayer_api.exchange_constants import EXCHANGES, CHAIN_GAS_TOKEN_MAP
from hummingbot.fluxlayer_api.get_chain_gas import get_btc_fee, get_gas_prices, get_solana_fee
from hummingbot.logger import HummingbotLogger

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

# 配置日志
_logger = HummingbotLogger(__name__)


# 全局变量 - 改为支持多个交易所
_exchanges = {}
_order_book_trackers = {}
_throttlers = {}
_sessions = {}
_initialized_exchanges = set()
_initialization_locks = {}


async def add_trading_pairs_to_exchange(exchange_name: str, new_trading_pairs: list):
    """向已初始化的交易所添加新的交易对"""
    global _exchanges, _order_book_trackers, _throttlers, _sessions, _initialized_exchanges, _initialization_locks
    
    if exchange_name not in _initialized_exchanges:
        _logger.warning(f"Exchange {exchange_name} not initialized, cannot add trading pairs")
        return False
        
    if exchange_name not in _initialization_locks:
        _initialization_locks[exchange_name] = asyncio.Lock()
        
    async with _initialization_locks[exchange_name]:
        try:
            order_book_tracker = _order_book_trackers[exchange_name]
            existing_pairs = set(order_book_tracker.order_books.keys())
            pairs_to_add = set(new_trading_pairs) - existing_pairs
            
            if not pairs_to_add:
                _logger.info(f"All trading pairs already exist for {exchange_name}")
                return True
                
            _logger.info(f"Adding new trading pairs {pairs_to_add} to {exchange_name}")
            
            # 获取数据源并添加新的交易对
            data_source = order_book_tracker._data_source
            if hasattr(data_source, '_trading_pairs'):
                # 更新数据源的交易对列表
                current_pairs = list(data_source._trading_pairs)
                updated_pairs = current_pairs + list(pairs_to_add)
                data_source._trading_pairs = updated_pairs
                
                # 重新启动数据源以包含新的交易对
                if hasattr(data_source, '_listen_for_order_book_snapshots'):
                    # 这里可能需要重启相关的监听器，但保持现有连接
                    _logger.info(f"Updated data source trading pairs for {exchange_name}")
                    
            return True
            
        except Exception as e:
            _logger.error(f"Error adding trading pairs to {exchange_name}: {e}")
            return False


async def initialize_exchange(exchange_name: str = "binance", trading_pairs: list = None):
    """初始化交易所连接 - 包含错误隔离机制"""
    global _exchanges, _order_book_trackers, _throttlers, _sessions, _initialized_exchanges, _initialization_locks

    _logger.info(f"🔧 [INIT DEBUG] Starting initialize_exchange for {exchange_name} with pairs: {trading_pairs}")

    try:
        if exchange_name not in EXCHANGES:
            _logger.error(f"❌ [INIT DEBUG] Unsupported exchange: {exchange_name}")
            return False

        # 为每个交易所创建独立的锁
        if exchange_name not in _initialization_locks:
            _initialization_locks[exchange_name] = asyncio.Lock()
    except Exception as e:
        _logger.error(f"❌ [INIT DEBUG] Pre-initialization error for {exchange_name}: {e}")
        return False

    try:
        exchange_config = EXCHANGES[exchange_name]
        _logger.info(f"🔧 [INIT DEBUG] Got exchange config for {exchange_name}")

        async with _initialization_locks[exchange_name]:
            # 检查是否需要初始化交易所或添加新的交易对
            new_trading_pairs = trading_pairs or ["BTC-USDT"]
            _logger.info(f"🔧 [INIT DEBUG] Processing trading pairs: {new_trading_pairs}")
            
            # 检查是否已经有这些交易对
            needs_initialization = False
            if exchange_name not in _initialized_exchanges:
                needs_initialization = True
                _logger.info(f"🔧 [INIT DEBUG] {exchange_name} not in initialized exchanges, needs initialization")
            elif exchange_name in _order_book_trackers:
                # 检查是否有新的交易对需要添加
                try:
                    # 尝试获取现有的交易对 - 使用order_books的keys
                    existing_pairs = set(_order_book_trackers[exchange_name].order_books.keys())
                    new_pairs = set(new_trading_pairs)
                    if not new_pairs.issubset(existing_pairs):
                        # 尝试增量添加而不是重新初始化
                        pairs_to_add = new_pairs - existing_pairs
                        _logger.info(f"Attempting to add new trading pairs {pairs_to_add} to {exchange_name}")
                        
                        success = await add_trading_pairs_to_exchange(exchange_name, list(pairs_to_add))
                        if success:
                            _logger.info(f"Successfully added new trading pairs to {exchange_name}")
                            return
                        else:
                            _logger.warning(f"Failed to add trading pairs incrementally, reinitializing {exchange_name}")
                            needs_initialization = True
                            # 清理旧的实例
                            if exchange_name in _order_book_trackers:
                                _order_book_trackers[exchange_name].stop()
                            if exchange_name in _sessions:
                                await _sessions[exchange_name].close()
                            _initialized_exchanges.discard(exchange_name)
                    else:
                        _logger.info(f"All requested trading pairs already available for {exchange_name}")
                        return
                except Exception as e:
                    _logger.warning(f"Error checking existing trading pairs for {exchange_name}: {e}")
                    # 如果无法检查，就重新初始化
                    needs_initialization = True
                    if exchange_name in _order_book_trackers:
                        _order_book_trackers[exchange_name].stop()
                    if exchange_name in _sessions:
                        await _sessions[exchange_name].close()
                    _initialized_exchanges.discard(exchange_name)
            
            if needs_initialization:
                try:
                    _logger.info(f"Initializing {exchange_name} exchange with trading pairs: {new_trading_pairs}")
                    
                    # 创建客户端配置
                    client_config = ClientConfigMap()
                    client_config.anonymized_metrics_mode = AnonymizedMetricsEnabledMode()
                    client_config_adapter = ClientConfigAdapter(client_config)

                    # 创建交易所实例
                    exchange_params = {
                        "client_config_map": client_config_adapter,
                        "trading_pairs": new_trading_pairs,
                        "trading_required": False
                    }
                    # 添加交易所特定的必要参数
                    exchange_params.update(exchange_config["required_params"])

                    exchange = exchange_config["exchange_class"](**exchange_params)

                    # 创建 throttler
                    throttler = AsyncThrottler(exchange_config["constants"].RATE_LIMITS)

                    proxy = os.environ.get("HTTP_PROXY")
                    # 创建带代理的 ClientSession
                    connector = TCPConnector(ssl=False)
                    session = ClientSession(
                        connector=connector,
                        timeout=ClientTimeout(total=30),
                        trust_env=True,
                        proxy=proxy
                    )

                    # 创建 WebAssistantsFactory
                    api_factory = WebAssistantsFactory(
                        throttler=throttler
                    )

                    # 创建订单簿数据源
                    data_source = exchange_config["data_source_class"](
                        trading_pairs=new_trading_pairs,
                        connector=exchange,
                        api_factory=api_factory
                    )

                    # 创建订单簿跟踪器
                    order_book_tracker = OrderBookTracker(
                        data_source=data_source,
                        trading_pairs=new_trading_pairs
                        )

                    # 启动订单簿跟踪器
                    _logger.info(f"🔧 [INIT DEBUG] Starting order book tracker for {exchange_name}")
                    order_book_tracker.start()

                    # 等待订单簿数据加载
                    _logger.info(f"🔧 [INIT DEBUG] Waiting for order book tracker to be ready for {exchange_name}")
                    await order_book_tracker.wait_ready()
                    _logger.info(f"✅ [INIT DEBUG] Order book tracker ready for {exchange_name}")

                    # 存储到全局字典
                    _exchanges[exchange_name] = exchange
                    _order_book_trackers[exchange_name] = order_book_tracker
                    _throttlers[exchange_name] = throttler
                    _sessions[exchange_name] = session
                    _initialized_exchanges.add(exchange_name)
                    
                    # 打印初始化后的状态
                    _logger.info(f"✅ [INIT DEBUG] Successfully initialized {exchange_name} exchange")
                    _logger.info(f"🔍 [INIT DEBUG] Trading pairs requested: {new_trading_pairs}")
                    _logger.info(f"🔍 [INIT DEBUG] OrderBookTracker created: {type(order_book_tracker)}")
                    
                    # 检查实际可用的订单簿
                    try:
                        if hasattr(order_book_tracker, 'order_books'):
                            available_pairs = list(order_book_tracker.order_books.keys())
                            _logger.info(f"✅ [INIT DEBUG] Available order books after initialization: {available_pairs}")
                            _logger.info(f"🔍 [INIT DEBUG] Order books count: {len(available_pairs)}")
                            
                            # 打印每个订单簿的详细信息
                            for pair in available_pairs:
                                order_book = order_book_tracker.order_books[pair]
                                _logger.info(f"🔍 [INIT DEBUG] - {pair}: {type(order_book)}")
                        else:
                            _logger.error(f"❌ [INIT DEBUG] OrderBookTracker has no order_books attribute")
                    except Exception as e:
                        _logger.error(f"❌ [INIT DEBUG] Error checking order books after initialization: {e}")
                except Exception as e:
                    _logger.error(f"❌ [INIT DEBUG] Failed to initialize exchange {exchange_name}: {e}")
                    # 清理可能的部分初始化状态
                    try:
                        if exchange_name in _order_book_trackers:
                            _order_book_trackers[exchange_name].stop()
                            del _order_book_trackers[exchange_name]
                        if exchange_name in _sessions:
                            await _sessions[exchange_name].close()
                            del _sessions[exchange_name]
                        if exchange_name in _exchanges:
                            del _exchanges[exchange_name]
                        if exchange_name in _throttlers:
                            del _throttlers[exchange_name]
                        _initialized_exchanges.discard(exchange_name)
                    except Exception as cleanup_error:
                        _logger.error(f"❌ [INIT DEBUG] Error during cleanup after failed initialization: {cleanup_error}")
                    return False
    except Exception as outer_e:
        _logger.error(f"❌ [INIT DEBUG] Outer exception in initialize_exchange for {exchange_name}: {outer_e}")
        return False


async def cleanup():
    """清理所有交易所资源"""
    global _exchanges, _order_book_trackers, _throttlers, _sessions, _initialized_exchanges, _initialization_locks

    try:
        # 获取所有已初始化的交易所列表
        exchanges_to_cleanup = list(_initialized_exchanges)
        
        _logger.info(f"Cleaning up {len(exchanges_to_cleanup)} exchanges: {exchanges_to_cleanup}")
        
        # 并行清理所有交易所
        cleanup_tasks = []
        for exchange_name in exchanges_to_cleanup:
            cleanup_tasks.append(cleanup_exchange(exchange_name))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # 确保所有全局状态都被清理
        _exchanges.clear()
        _order_book_trackers.clear()
        _throttlers.clear()
        _sessions.clear()
        _initialized_exchanges.clear()
        _initialization_locks.clear()
        
        _logger.info("Successfully cleaned up all exchange resources")
    except Exception as e:
        _logger.error(f"Error during global cleanup: {e}")
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


async def get_supported_trading_pairs(exchange_name: str) -> list:
    """获取交易所支持的交易对列表"""
    if exchange_name == "hyperliquid":
        # Hyperliquid 支持的交易对 - 基于API调查结果更新
        return [
            "UBTC-USDC",  # BTC -> UBTC-USDC
            "UETH-USDC",  # ETH -> UETH-USDC  
            "USDT0-USDC"  # USDC-USDT -> USDT0-USDC (用于USDC/USDT汇率转换)
        ]
    elif exchange_name == "bybit":
        # Bybit 支持的标准交易对 - 移除可能不存在的 USDC-USDT
        return ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    elif exchange_name == "okx":
        # OKX 支持的标准交易对 - 移除可能不存在的 USDC-USDT
        return ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    else:
        # 默认交易对 - 只包含标准的主流交易对
        return ["BTC-USDT", "ETH-USDT", "SOL-USDT"]


async def validate_trading_pair(exchange_name: str, symbol: str) -> bool:
    """验证交易对是否被交易所支持"""
    supported_pairs = await get_supported_trading_pairs(exchange_name)
    return symbol in supported_pairs


async def get_order_book(exchange_name: str, token: str):
    """获取代币的订单簿数据 - 包含错误隔离机制"""
    from hummingbot.fluxlayer_api.exchange_constants import get_trading_symbol, validate_hyperliquid_token, is_stablecoin
    
    _logger.info(f"🚀 [ORDER BOOK DEBUG] Starting get_order_book for exchange={exchange_name}, token={token}")
    
    # 检查是否为稳定币 - 稳定币无需查询交易所
    if is_stablecoin(token):
        _logger.info(f"💰 [STABLECOIN DEBUG] {token} is a stablecoin, skipping exchange query")
        return None  # 返回 None 表示这是稳定币，调用方会特殊处理
    
    try:
        # 预先验证 Hyperliquid 代币支持
        if exchange_name == "hyperliquid" and not validate_hyperliquid_token(token):
            _logger.warning(f"❌ [ORDER BOOK DEBUG] Token {token} not supported on Hyperliquid")
            return None
            
        symbol = get_trading_symbol(exchange_name, token)
        _logger.info(f"✅ [ORDER BOOK DEBUG] Symbol mapping: {exchange_name} - {token} -> {symbol}")
    except ValueError as e:
        _logger.error(f"❌ [ORDER BOOK DEBUG] Invalid trading symbol for {exchange_name} - {token}: {e}")
        return None
    except Exception as e:
        _logger.error(f"❌ [ORDER BOOK DEBUG] Unexpected error in symbol mapping for {exchange_name}: {e}")
        return None
    
    try:
        # 获取交易所支持的交易对列表
        supported_pairs = await get_supported_trading_pairs(exchange_name)
        _logger.info(f"📋 [ORDER BOOK DEBUG] {exchange_name} supported pairs: {supported_pairs}")
        
        # 验证交易对是否被支持
        is_supported = await validate_trading_pair(exchange_name, symbol)
        _logger.info(f"🔍 [ORDER BOOK DEBUG] Symbol {symbol} supported on {exchange_name}: {is_supported}")
        
        if not is_supported:
            _logger.warning(f"❌ [ORDER BOOK DEBUG] Trading pair {symbol} not supported on {exchange_name}")
            return None
        
        # 确保当前需要的交易对在列表中
        if symbol not in supported_pairs:
            supported_pairs.append(symbol)
            _logger.info(f"➕ [ORDER BOOK DEBUG] Added {symbol} to supported pairs: {supported_pairs}")
        
        # 确保交易所已初始化
        _logger.info(f"🔧 [ORDER BOOK DEBUG] Initializing exchange {exchange_name} with pairs: {supported_pairs}")
        await initialize_exchange(exchange_name, supported_pairs)

        # 检查交易所是否已初始化
        if exchange_name not in _order_book_trackers:
            _logger.error(f"❌ [ORDER BOOK DEBUG] Exchange {exchange_name} not initialized. Available: {list(_order_book_trackers.keys())}")
            return None
            
        order_book_tracker = _order_book_trackers[exchange_name]
        _logger.info(f"✅ [ORDER BOOK DEBUG] Got order book tracker for {exchange_name}")
        
        # 打印 OrderBookTracker 的详细信息
        _logger.info(f"🔍 [DEBUG] OrderBookTracker for {exchange_name}:")
        _logger.info(f"🔍 [DEBUG] - Type: {type(order_book_tracker)}")
        _logger.info(f"🔍 [DEBUG] - Dir: {[attr for attr in dir(order_book_tracker) if not attr.startswith('_')]}")
        
        try:
            _logger.info(f"🔍 [DEBUG] - order_books type: {type(order_book_tracker.order_books)}")
            _logger.info(f"🔍 [DEBUG] - order_books keys: {list(order_book_tracker.order_books.keys())}")
            _logger.info(f"🔍 [DEBUG] - order_books length: {len(order_book_tracker.order_books)}")
        except Exception as e:
            _logger.error(f"🔍 [DEBUG] - Error accessing order_books: {e}")
        
        # 检查是否有其他可能的属性
        for attr_name in ['trading_pairs', '_trading_pairs', 'data_source']:
            if hasattr(order_book_tracker, attr_name):
                try:
                    attr_value = getattr(order_book_tracker, attr_name)
                    _logger.info(f"🔍 [DEBUG] - {attr_name}: {attr_value} (type: {type(attr_value)})")
                except Exception as e:
                    _logger.error(f"🔍 [DEBUG] - Error accessing {attr_name}: {e}")
        
        # 检查交易对是否存在于订单簿中
        available_symbols = list(order_book_tracker.order_books.keys())
        _logger.info(f"🔍 [ORDER BOOK DEBUG] Available symbols in {exchange_name}: {available_symbols}")
        
        if symbol not in order_book_tracker.order_books:
            _logger.error(f"❌ [ORDER BOOK DEBUG] Symbol {symbol} not found in order books for {exchange_name}. Available symbols: {available_symbols}")
            _logger.error(f"🔍 [ORDER BOOK DEBUG] Requested symbol: '{symbol}' (length: {len(symbol)})")
            for avail_symbol in available_symbols:
                _logger.error(f"🔍 [ORDER BOOK DEBUG] Available: '{avail_symbol}' (length: {len(avail_symbol)})")
            return None

        # 获取订单簿数据
        order_book = order_book_tracker.order_books[symbol]
        _logger.info(f"✅ [ORDER BOOK DEBUG] Got order book for {symbol}")
        
        # 打印订单簿详细信息
        _logger.info(f"🔍 [DEBUG] OrderBook for {symbol} on {exchange_name}:")
        _logger.info(f"🔍 [DEBUG] - OrderBook type: {type(order_book)}")
        try:
            bid_entries = list(order_book.bid_entries())
            ask_entries = list(order_book.ask_entries())
            _logger.info(f"🔍 [DEBUG] - Bid entries count: {len(bid_entries)}")
            _logger.info(f"🔍 [DEBUG] - Ask entries count: {len(ask_entries)}")
            if bid_entries:
                _logger.info(f"🔍 [DEBUG] - Top bid: {bid_entries[0]}")
            if ask_entries:
                _logger.info(f"🔍 [DEBUG] - Top ask: {ask_entries[0]}")
        except Exception as e:
            _logger.error(f"🔍 [DEBUG] - Error accessing order book entries: {e}")

        # 获取买卖单 - 确保转换为 list 格式
        try:
            bid_entries = list(order_book.bid_entries())
            ask_entries = list(order_book.ask_entries())
            
            bids = [(str(price), str(amount)) for price, amount, _ in bid_entries]
            asks = [(str(price), str(amount)) for price, amount, _ in ask_entries]
        except Exception as e:
            _logger.error(f"Error converting order book entries to list format: {e}")
            return None
        
        # 检查订单簿是否有数据
        if not bids or not asks:
            _logger.warning(f"Order book for {symbol} on {exchange_name} has no data. Bids: {len(bids)}, Asks: {len(asks)}")
            return None
            
        _logger.info(f"Successfully got order book for {symbol} on {exchange_name}. Bids: {len(bids)}, Asks: {len(asks)}")
        return {
            'lastUpdateId': int(time.time() * 1000),
            'bids': bids,
            'asks': asks
        }
    except KeyError as e:
        _logger.error(f"KeyError getting order book for {symbol} on {exchange_name}: {e}")
        return None
    except Exception as e:
        _logger.error(f"Error getting order book for {token} on {exchange_name} (symbol: {symbol}): {e}")
        # 发生错误时，尝试清理可能损坏的连接
        try:
            if exchange_name in _initialized_exchanges and exchange_name in _order_book_trackers:
                # 检查连接是否仍然有效
                order_book_tracker = _order_book_trackers[exchange_name]
                if not hasattr(order_book_tracker, 'order_books') or len(order_book_tracker.order_books) == 0:
                    _logger.warning(f"Order book tracker for {exchange_name} appears corrupted, reinitializing...")
                    await cleanup_exchange(exchange_name)
        except Exception as cleanup_error:
            _logger.error(f"Error during cleanup attempt: {cleanup_error}")
        return None


async def cleanup_exchange(exchange_name: str):
    """清理指定交易所的资源"""
    global _exchanges, _order_book_trackers, _throttlers, _sessions, _initialized_exchanges, _initialization_locks
    
    if exchange_name not in _initialization_locks:
        _initialization_locks[exchange_name] = asyncio.Lock()
        
    async with _initialization_locks[exchange_name]:
        try:
            _logger.info(f"Cleaning up resources for {exchange_name}")
            
            if exchange_name in _order_book_trackers:
                try:
                    _order_book_trackers[exchange_name].stop()
                except Exception as e:
                    _logger.warning(f"Error stopping order book tracker for {exchange_name}: {e}")
                del _order_book_trackers[exchange_name]
                
            if exchange_name in _sessions:
                try:
                    await _sessions[exchange_name].close()
                except Exception as e:
                    _logger.warning(f"Error closing session for {exchange_name}: {e}")
                del _sessions[exchange_name]
                
            if exchange_name in _exchanges:
                del _exchanges[exchange_name]
                
            if exchange_name in _throttlers:
                del _throttlers[exchange_name]
                
            _initialized_exchanges.discard(exchange_name)
            
            _logger.info(f"Successfully cleaned up resources for {exchange_name}")
            
        except Exception as e:
            _logger.error(f"Error during cleanup of {exchange_name}: {e}")
            raise


async def get_quote_price(exchange_name, trading_pair: str, is_buy: bool, quote_volume):
    """获取交易对的报价"""
    try:
        # TODO 这里初始化交易所的方式可能需要改进
        await initialize_exchange(exchange_name, [trading_pair])
        order_book_tracker = _order_book_trackers[exchange_name]
        order_book = order_book_tracker.order_books[trading_pair]
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
        # data = get_solana_fee()
        return "SOL", 0.000105
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
    """
    获取单个交易所的RFQ结果 - 基于正确的交易对逻辑重写
    
    核心理解:
    - 交易对格式: BASE-QUOTE (如 BTC-USDC)
    - is_buy=True: 用 QUOTE 买 BASE (用 USDC 买 BTC)
    - is_buy=False: 卖 BASE 换 QUOTE (卖 BTC 换 USDC)
    - target_amount: 目标代币的数量
    - target_price: 目标代币的价格
    """
    try:
        from hummingbot.fluxlayer_api.exchange_constants import is_stablecoin, get_stablecoin_price, get_trading_symbol
        
        _logger.info(f"🔄 [RFQ DEBUG] Starting RFQ for {exchange_name}: {source_token} -> {target_token}, is_buy={is_buy}, amount={amount}")
        
        # 参数验证
        if amount <= 0:
            _logger.warning(f"❌ [RFQ DEBUG] Invalid amount {amount} for {exchange_name}")
            return None
        
        # 特殊情况: 稳定币之间的1:1兑换
        if is_stablecoin(source_token) and is_stablecoin(target_token):
            _logger.info(f"💰 [RFQ DEBUG] Stablecoin to stablecoin: {source_token} -> {target_token} (1:1)")
            return {
                "exchange": exchange_name,
                "source_amount": amount,
                "source_price": get_stablecoin_price(),
                "target_amount": amount,  # 1:1 兑换
                "target_price": get_stablecoin_price(),
            }
        
        # 构建交易对: SOURCE-TARGET
        # 但是要注意交易对的方向和 is_buy 参数的含义
        _logger.info(f"🎯 [RFQ DEBUG] Building trading pair: {source_token}-{target_token}")
        
        try:
            # 尝试构建 source_token -> target_token 的交易对
            trading_pair = get_trading_symbol(exchange_name, source_token, target_token)
            _logger.info(f"✅ [RFQ DEBUG] Trading pair: {trading_pair}")
            
            # 获取订单簿数据 - 这里是关键，要获取正确的交易对
            order_book = await get_order_book(exchange_name, source_token)
            if not order_book:
                _logger.warning(f"❌ [RFQ DEBUG] Failed to get order book for {source_token} on {exchange_name}")
                return None
            
            # 解析交易逻辑
            if is_buy:
                # is_buy=True: 用 amount 金额买 source_token
                # 例如: BTC-USDC, amount=10000, is_buy=True -> 用10000U买BTC
                # amount 是投入的资金（target_token），要计算能买到多少 source_token
                ask_price = float(order_book['asks'][0][0])
                _logger.info(f"💹 [RFQ DEBUG] Buy operation: Using ask price {ask_price} for {source_token}")
                
                # amount 资金能买到多少 source_token
                target_amount_received = amount / ask_price  # 投入金额 ÷ 代币价格 = 能买到的代币数量
                
                return {
                    "exchange": exchange_name,
                    "source_amount": amount,  # 投入的资金金额
                    "source_price": ask_price,  # source_token的单价
                    "target_amount": target_amount_received,  # 能买到的source_token数量
                    "target_price": ask_price,  # source_token的单价
                }
            else:
                # is_buy=False: 卖出 source_token 获得 amount 金额
                # 例如: BTC-USDC, amount=10000, is_buy=False -> 卖BTC获得10000U
                # amount 是期望获得的资金（target_token），要计算需要卖多少 source_token
                bid_price = float(order_book['bids'][0][0])
                _logger.info(f"💹 [RFQ DEBUG] Sell operation: Using bid price {bid_price} for {source_token}")
                
                # 要获得 amount 资金需要卖多少 source_token
                target_amount_needed = amount / bid_price  # 期望金额 ÷ 代币价格 = 需要卖出的代币数量
                
                return {
                    "exchange": exchange_name,
                    "source_amount": amount,  # 期望获得的资金金额
                    "source_price": bid_price,  # source_token的单价
                    "target_amount": target_amount_needed,  # 需要卖出的source_token数量
                    "target_price": bid_price,  # source_token的单价
                }
                
        except ValueError as e:
            # 如果 source-target 交易对无效，尝试反向交易对
            _logger.warning(f"⚠️ [RFQ DEBUG] Direct trading pair failed: {e}, trying reverse pair")
            
            try:
                # 尝试构建 target_token -> source_token 的交易对
                reverse_trading_pair = get_trading_symbol(exchange_name, target_token, source_token)
                _logger.info(f"🔄 [RFQ DEBUG] Reverse trading pair: {reverse_trading_pair}")
                
                # 获取反向交易对的订单簿
                reverse_order_book = await get_order_book(exchange_name, target_token)
                if not reverse_order_book:
                    _logger.warning(f"❌ [RFQ DEBUG] Failed to get reverse order book for {target_token} on {exchange_name}")
                    return None
                
                # 反向交易逻辑
                if is_buy:
                    # 原本是用 target 买 source，现在用反向交易对实现
                    # 在反向交易对中，这相当于卖 target 换 source，使用 bid 价格
                    reverse_bid_price = float(reverse_order_book['bids'][0][0])
                    source_price = 1.0 / reverse_bid_price  # source_token 相对于 target_token 的价格
                    target_amount_needed = amount * source_price
                    
                    return {
                        "exchange": exchange_name,
                        "source_amount": amount,
                        "source_price": source_price,
                        "target_amount": target_amount_needed,
                        "target_price": source_price,  # source_token 相对于 target_token 的价格
                    }
                else:
                    # 原本是卖 source 换 target，现在用反向交易对实现
                    # 在反向交易对中，这相当于买 target 用 source，使用 ask 价格
                    reverse_ask_price = float(reverse_order_book['asks'][0][0])
                    source_price = 1.0 / reverse_ask_price  # source_token 相对于 target_token 的价格
                    target_amount_received = amount * source_price
                    
                    return {
                        "exchange": exchange_name,
                        "source_amount": amount,
                        "source_price": source_price,
                        "target_amount": target_amount_received,
                        "target_price": source_price,  # source_token 相对于 target_token 的价格
                    }
                    
            except ValueError as reverse_e:
                _logger.error(f"❌ [RFQ DEBUG] Both direct and reverse trading pairs failed: {reverse_e}")
                return None
            
    except Exception as e:
        _logger.error(f"❌ [RFQ DEBUG] Critical error in {exchange_name} RFQ: {e}")
        import traceback
        _logger.error(f"❌ [RFQ DEBUG] Stack trace for {exchange_name}: {traceback.format_exc()}")
        return None

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
    """并行查询多个交易所并返回最优报价 - 包含错误隔离机制"""
    try:
        _logger.info(f"🚀 [MULTI-RFQ DEBUG] Starting multi-exchange RFQ: {source_token} -> {target_token}")
        
        # 参数验证
        if amount <= 0:
            return {"error": "Amount must be greater than 0"}
        if not all([source_chain, source_token, target_chain, target_token]):
            return {"error": "Missing required parameters"}

        # 并行查询所有交易所，使用return_exceptions=True来隔离错误
        tasks = [
            get_single_exchange_rfq(
                source_chain, source_token, amount,
                target_chain, target_token, exchange, is_buy
            )
            for exchange in EXCHANGES.keys()
        ]

        _logger.info(f"🔄 [MULTI-RFQ DEBUG] Starting {len(tasks)} parallel exchange requests")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分析结果，区分正常结果和异常
        valid_results = []
        error_count = 0
        
        for i, result in enumerate(results):
            exchange_name = list(EXCHANGES.keys())[i]
            
            if isinstance(result, Exception):
                _logger.error(f"❌ [MULTI-RFQ DEBUG] Exception from {exchange_name}: {result}")
                error_count += 1
            elif result is None:
                _logger.warning(f"⚠️ [MULTI-RFQ DEBUG] No result from {exchange_name}")
                error_count += 1
            elif isinstance(result, dict) and "target_amount" in result:
                _logger.info(f"✅ [MULTI-RFQ DEBUG] Valid result from {exchange_name}: {result['target_amount']}")
                valid_results.append(result)
            else:
                _logger.warning(f"⚠️ [MULTI-RFQ DEBUG] Invalid result format from {exchange_name}: {type(result)}")
                error_count += 1
        
        _logger.info(f"📊 [MULTI-RFQ DEBUG] Results summary: {len(valid_results)} valid, {error_count} errors")

        if not valid_results:
            _logger.error(f"❌ [MULTI-RFQ DEBUG] No valid quotes from any exchange")
            return {"error": f"No valid quotes available from {len(EXCHANGES)} exchanges"}

        best_result = max(valid_results, key=lambda x: float(x["target_amount"]))
        best_exchange = best_result["exchange"]
        _logger.info(f"🏆 [MULTI-RFQ DEBUG] Best quote from {best_exchange}: {best_result['target_amount']}")
        
        best_result["target_amount"] = truncate_amount(str(best_result["target_amount"]), target_token)

        # 统一返回格式，包含详细的交易所状态信息
        all_exchanges_info = {}
        for r in valid_results:
            all_exchanges_info[r["exchange"]] = {
                "target_amount": r["target_amount"],
                "target_price": r["target_price"],
                "status": "success"
            }
        
        # 添加失败的交易所信息
        for i, result in enumerate(results):
            exchange_name = list(EXCHANGES.keys())[i]
            if exchange_name not in all_exchanges_info:
                if isinstance(result, Exception):
                    status = f"error: {str(result)[:100]}"  # 限制错误信息长度
                else:
                    status = "no_data"
                all_exchanges_info[exchange_name] = {
                    "target_amount": None,
                    "target_price": None,
                    "status": status
                }
        
        best_result["all_exchanges"] = all_exchanges_info
        best_result["summary"] = {
            "valid_exchanges": len(valid_results),
            "total_exchanges": len(EXCHANGES),
            "success_rate": f"{len(valid_results)/len(EXCHANGES)*100:.1f}%"
        }

        return best_result
    except Exception as e:
        _logger.error(f"❌ [MULTI-RFQ DEBUG] Critical error in get_best_rfq: {e}")
        import traceback
        _logger.error(f"❌ [MULTI-RFQ DEBUG] Stack trace: {traceback.format_exc()}")
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


async def test_multi_exchange_concurrent():
    """测试多交易所并发运行"""
    _logger.info("=== Testing Multi-Exchange Concurrent Operations ===")
    
    try:
        # 测试不同交易所的并发初始化和订单簿获取
        test_exchanges = ["bybit", "okx", "hyperliquid"]
        
        tasks = []
        for exchange in test_exchanges:
            if exchange == "hyperliquid":
                # Hyperliquid 只测试确认支持的代币
                for token in ["BTC"]:  # 只测试 BTC->UBTC-USDC
                    tasks.append(get_order_book(exchange, token))
            else:
                # 其他交易所测试标准代币
                for token in ["BTC", "ETH"]:  # 只测试主流代币
                    tasks.append(get_order_book(exchange, token))
        
        _logger.info(f"Starting {len(tasks)} concurrent order book requests...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分析结果
        success_count = 0
        error_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                _logger.error(f"Task {i} failed: {result}")
                error_count += 1
            elif result is None:
                _logger.warning(f"Task {i} returned None")
                error_count += 1
            elif isinstance(result, dict):
                bids_count = len(result.get('bids', []))
                asks_count = len(result.get('asks', []))
                _logger.info(f"Task {i} succeeded: {bids_count} bids, {asks_count} asks")
                success_count += 1
            else:
                _logger.warning(f"Task {i} returned unexpected type: {type(result)}")
                error_count += 1
        
        _logger.info(f"Test completed: {success_count} successes, {error_count} errors")
        
        # 检查全局状态
        _logger.info(f"Initialized exchanges: {list(_initialized_exchanges)}")
        _logger.info(f"Active order book trackers: {list(_order_book_trackers.keys())}")
        
        return success_count > 0 and success_count >= error_count
        
    except Exception as e:
        _logger.error(f"Test failed with exception: {e}")
        return False
    finally:
        # 清理测试资源
        try:
            await cleanup()
            _logger.info("Test cleanup completed")
        except Exception as cleanup_error:
            _logger.error(f"Test cleanup failed: {cleanup_error}")


if __name__ == "__main__":
    # 简单的测试入口
    async def main():
        print("RFQ Multi-Exchange Test")
        success = await test_multi_exchange_concurrent()
        print(f"Test {'PASSED' if success else 'FAILED'}")
    
    asyncio.run(main())