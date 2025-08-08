"""
FluxLayer Order Manager

面向对象的订单管理类，提供下单和查询订单状态的功能
"""
import asyncio
import os
import time
from decimal import Decimal
from typing import Dict, Any, Optional

from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.core.data_type.common import OrderType
from hummingbot.logger import HummingbotLogger
from hummingbot.fluxlayer_api.exchange_constants import EXCHANGES, get_supported_exchanges, get_exchange_config, validate_exchange_support


class OrderManager:
    """FluxLayer 订单管理器"""
    
    def __init__(self, start_background_initialization: bool = False):
        self._logger = HummingbotLogger(__name__)
        self._connectors: Dict[str, Any] = {}
        self._connector_locks: Dict[str, asyncio.Lock] = {}
        self._initialization_task: Optional[asyncio.Task] = None
        self._is_initializing = False
        self._initialization_complete = False
        
        # 如果启用了自动初始化，启动后台初始化任务
        if start_background_initialization:
            self.start_background_initialization()
        
    async def initialize_all_connectors(self, default_trading_pairs: Dict[str, str] = None) -> Dict[str, bool]:
        """
        预初始化所有支持的连接器
        
        参数:
            default_trading_pairs: 默认交易对映射 {"exchange_name": "trading_pair"}
                                   如果未提供，将使用通用交易对
        
        返回:
            Dict[str, bool]: 各交易所初始化结果 {"exchange_name": success}
        """
        if self._is_initializing:
            # 如果正在初始化，等待完成
            if self._initialization_task:
                await self._initialization_task
            return {name: name in self._connectors for name in get_supported_exchanges()}
        
        if self._initialization_complete:
            # 已经完成初始化
            return {name: name in self._connectors for name in get_supported_exchanges()}
        
        self._is_initializing = True
        
        # 默认交易对配置
        if default_trading_pairs is None:
            default_trading_pairs = {
                "binance": "BTC-USDT",
                "bybit": "BTC-USDT", 
                "okx": "BTC-USDT",
                "hyperliquid": "UBTC-USDC"  # Hyperliquid 现货市场使用 UBTC
            }
        
        supported_exchanges = get_supported_exchanges()
        initialization_results = {}
        
        self._logger.info(f"Starting pre-initialization of {len(supported_exchanges)} connectors...")
        
        # 并行初始化所有连接器
        initialization_tasks = []
        for exchange_name in supported_exchanges:
            trading_pair = default_trading_pairs.get(exchange_name, "BTC-USDT")
            task = self._initialize_single_connector(exchange_name, trading_pair)
            initialization_tasks.append(task)
        
        # 等待所有初始化任务完成
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # 处理结果
        for exchange_name, result in zip(supported_exchanges, results):
            if isinstance(result, Exception):
                self._logger.error(f"Failed to initialize {exchange_name}: {result}")
                initialization_results[exchange_name] = False
            else:
                initialization_results[exchange_name] = result
                if result:
                    self._logger.info(f"Successfully initialized {exchange_name}")
                else:
                    self._logger.warning(f"Failed to initialize {exchange_name}")
        
        successful_count = sum(initialization_results.values())
        self._logger.info(f"Pre-initialization complete: {successful_count}/{len(supported_exchanges)} connectors ready")
        
        self._is_initializing = False
        self._initialization_complete = True
        return initialization_results
    
    async def _initialize_single_connector(self, connector_name: str, trading_pair: str) -> bool:
        """
        初始化单个连接器
        
        参数:
            connector_name: 交易所名称
            trading_pair: 交易对
            
        返回:
            bool: 初始化是否成功
        """
        try:
            # 检查是否已经缓存
            if connector_name in self._connectors:
                return True
            
            # 验证交易所支持
            if not validate_exchange_support(connector_name):
                self._logger.warning(f"Exchange {connector_name} not supported")
                return False
            
            # 获取配置
            exchange_config = get_exchange_config(connector_name)
            required_params = exchange_config["required_params"]
            
            # 检查API密钥
            missing_params = self._validate_required_params(required_params, connector_name)
            if missing_params:
                self._logger.warning(f"Missing API credentials for {connector_name}: {missing_params}. Skipping initialization.")
                return False
            
            # 创建连接器
            connector = await self._create_connector_instance(connector_name, trading_pair, exchange_config, required_params)
            
            if connector:
                # 缓存连接器
                self._connectors[connector_name] = connector
                return True
            else:
                return False
                
        except Exception as e:
            self._logger.error(f"Error initializing connector {connector_name}: {e}")
            return False
    
    async def _create_connector_instance(self, connector_name: str, trading_pair: str, exchange_config: dict, required_params: dict):
        """
        创建连接器实例
        
        参数:
            connector_name: 交易所名称
            trading_pair: 交易对
            exchange_config: 交易所配置
            required_params: 必需参数
            
        返回:
            连接器实例或None
        """
        try:
            # 创建客户端配置
            client_config = ClientConfigMap()
            client_config_adapter = ClientConfigAdapter(client_config)
            
            # 准备交易所连接参数
            init_params = {
                "client_config_map": client_config_adapter,
                "trading_pairs": [trading_pair],
                "trading_required": True
            }
            
            # 添加交易所特定参数
            init_params.update(required_params)
            
            # 创建连接器实例
            exchange_class = exchange_config["exchange_class"]
            
            # 🔍 DEBUG: 记录连接器初始化参数
            if connector_name == "bybit":
                self._logger.info(f"🔍 [BYBIT INIT DEBUG] Creating {connector_name} with params:")
                for key, value in init_params.items():
                    if 'key' in key.lower() or 'secret' in key.lower():
                        masked_value = f"{str(value)[:8]}..." if len(str(value)) > 8 else "TOO_SHORT"
                        self._logger.info(f"🔍 [BYBIT INIT DEBUG]   {key}: {masked_value}")
                    else:
                        self._logger.info(f"🔍 [BYBIT INIT DEBUG]   {key}: {value}")
            
            connector = exchange_class(**init_params)
            
            # 🔍 DEBUG: 检查连接器创建后的状态
            if connector_name == "bybit":
                self._logger.info(f"🔍 [BYBIT INIT DEBUG] Connector created successfully")
                
                # 检查是否有必要的属性

                attrs_to_check = ['_web_assistants_factory', '_auth', '_throttler', '_time_synchronizer']
                for attr in attrs_to_check:
                    has_attr = hasattr(connector, attr)
                    attr_value = getattr(connector, attr, None) if has_attr else None
                    self._logger.info(f"🔍 [BYBIT INIT DEBUG] {attr}: {'EXISTS' if has_attr else 'MISSING'} (value: {'SET' if attr_value else 'NONE'})")
            
            # 启动网络连接
            await connector.start_network()
            
            # 🔍 DEBUG: 为 Hyperliquid 手动触发余额更新
            if connector_name == "hyperliquid":
                self._logger.info(f"🔍 [HYPERLIQUID DEBUG] Manually triggering balance update for {connector_name}")
                try:
                    if hasattr(connector, '_update_balances'):
                        await connector._update_balances()
                        self._logger.info(f"✅ [HYPERLIQUID DEBUG] Manual balance update completed")
                    else:
                        self._logger.warning(f"⚠️ [HYPERLIQUID DEBUG] No _update_balances method found")
                except Exception as e:
                    self._logger.error(f"❌ [HYPERLIQUID DEBUG] Manual balance update failed: {e}")
            
            # 等待基本状态就绪（较短超时，避免阻塞太久）
            max_wait_time = 15  # 15秒超时
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # 检查账户余额是否加载
                if len(connector._account_balances) > 0:
                    self._logger.debug(f"Account balance loaded for {connector_name}")
                    break
                await asyncio.sleep(1)
            
            # 尝试更新交易规则（快速失败）
            try:
                await asyncio.wait_for(connector._update_trading_rules(), timeout=10)
                self._logger.debug(f"Trading rules loaded for {connector_name}")
            except asyncio.TimeoutError:
                self._logger.warning(f"Trading rules update timeout for {connector_name}, continuing anyway")
            except Exception as e:
                self._logger.warning(f"Failed to update trading rules for {connector_name}: {e}, continuing anyway")
            
            return connector
            
        except Exception as e:
            self._logger.error(f"Failed to create connector instance for {connector_name}: {e}")
            return None
    
    def start_background_initialization(self, default_trading_pairs: Dict[str, str] = None):
        """
        启动后台初始化任务（非阻塞）
        
        参数:
            default_trading_pairs: 默认交易对映射
        """
        if not self._is_initializing and not self._initialization_complete:
            try:
                # 尝试获取当前事件循环
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，直接创建任务
                    self._initialization_task = asyncio.create_task(
                        self.initialize_all_connectors(default_trading_pairs)
                    )
                    self._logger.info("Background connector initialization started")
                else:
                    # 如果事件循环未运行，稍后手动调用
                    self._logger.info("Event loop not running, background initialization will start when needed")
            except RuntimeError:
                # 没有活动的事件循环，稍后手动调用
                self._logger.info("No active event loop found, background initialization will start when needed")

    def _validate_required_params(self, required_params: dict, connector_name: str) -> list:
        """
        验证必需参数是否提供，返回缺失的参数列表
        
        参数:
            required_params (dict): 必需参数字典
            connector_name (str): 连接器名称（用于日志）
            
        返回:
            list: 缺失的参数名称列表
        """
        missing_params = []
        for key, value in required_params.items():
            # 根据参数类型判断是否缺失
            if isinstance(value, bool):
                # 布尔参数：False 和 True 都是有效值，不视为缺失
                continue
            elif isinstance(value, str):
                # 字符串参数：空字符串视为缺失
                if not value:
                    missing_params.append(key)
            elif isinstance(value, (int, float)):
                # 数值参数：0 可能是有效值，只有 None 视为缺失
                if value is None:
                    missing_params.append(key)
            elif value is None:
                # None 值视为缺失
                missing_params.append(key)
            # 对于其他类型（如列表、字典等），只有 None 或空容器才视为缺失
            elif hasattr(value, '__len__') and len(value) == 0:
                # 空容器（如空列表、空字典）视为缺失
                missing_params.append(key)
        
        return missing_params

    async def _get_or_create_connector(self, connector_name: str, trading_pair: str, wait_for_orderbook: bool = False):
        """
        获取或创建连接器实例（优先使用预缓存的连接器）
        
        参数:
            connector_name (str): 交易所名称
            trading_pair (str): 交易对
            wait_for_orderbook (bool): 是否等待订单簿数据初始化完成
            
        返回:
            连接器实例
        """
        # 为每个交易所创建独立的锁
        if connector_name not in self._connector_locks:
            self._connector_locks[connector_name] = asyncio.Lock()
        
        async with self._connector_locks[connector_name]:
            # 检查指定的连接器是否已经在缓存中
            if connector_name in self._connectors:
                connector = self._connectors[connector_name]
                # 检查连接器是否仍然有效
                try:
                    if hasattr(connector, 'ready') and connector.ready:
                        self._logger.info(f"Using cached connector for {connector_name}")
                        return connector
                    elif hasattr(connector, '_account_balances') and len(connector._account_balances) > 0:
                        # 即使 ready 状态不完整，如果有账户余额，也可以使用
                        self._logger.info(f"Using cached connector for {connector_name} (partial ready state)")
                        return connector
                except Exception as e:
                    self._logger.warning(f"Cached connector validation failed for {connector_name}: {e}")
                    # 移除无效的连接器
                    del self._connectors[connector_name]
            
            # 如果正在进行预初始化，等待一小段时间再检查这个特定的连接器
            if self._is_initializing and connector_name not in self._connectors:
                self._logger.info(f"Pre-initialization in progress, waiting for {connector_name}...")
                await asyncio.sleep(2)  # 等待2秒
                
                # 再次检查这个特定连接器是否完成初始化
                if connector_name in self._connectors:
                    connector = self._connectors[connector_name]
                    try:
                        if (hasattr(connector, 'ready') and connector.ready) or \
                           (hasattr(connector, '_account_balances') and len(connector._account_balances) > 0):
                            self._logger.info(f"Using connector initialized during background process for {connector_name}")
                            return connector
                    except Exception as e:
                        self._logger.warning(f"Background initialized connector validation failed for {connector_name}: {e}")
                        # 移除无效连接器，继续创建新的
                        del self._connectors[connector_name]
            
            # 创建新的连接器（如果缓存中没有或无效）
            self._logger.info(f"Creating new connector for {connector_name}")
            
            if not validate_exchange_support(connector_name):
                raise ValueError(f"Unsupported exchange: {connector_name}")
            
            exchange_config = get_exchange_config(connector_name)
            
            # 检查API密钥是否提供
            required_params = exchange_config["required_params"]
            missing_params = self._validate_required_params(required_params, connector_name)
            if missing_params:
                raise ValueError(f"Missing required API credentials for {connector_name}: {missing_params}")
                
            # 对于 Bybit，记录 API 密钥信息（但不严格验证长度）
            if connector_name == "bybit":
                api_key = required_params.get("bybit_api_key", "")
                api_secret = required_params.get("bybit_api_secret", "")
                
                self._logger.info(f"🔍 [BYBIT VALIDATION] API key length: {len(api_key)}, secret length: {len(api_secret)}")
                self._logger.info(f"🔍 [BYBIT VALIDATION] API key prefix: {api_key[:8]}...")
                self._logger.info(f"🔍 [BYBIT VALIDATION] API secret prefix: {api_secret[:8]}...")
                
                # 只做基本检查
                if len(api_key) < 10:
                    self._logger.warning(f"⚠️ [BYBIT VALIDATION] API key seems unusually short: {len(api_key)} characters")
                if len(api_secret) < 20:
                    self._logger.warning(f"⚠️ [BYBIT VALIDATION] API secret seems unusually short: {len(api_secret)} characters")
            
            # 创建客户端配置
            client_config = ClientConfigMap()
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
            
            # 🔍 DEBUG: 记录连接器初始化参数（主方法）
            if connector_name == "bybit":
                self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] Creating {connector_name} with params:")
                for key, value in init_params.items():
                    if 'key' in key.lower() or 'secret' in key.lower():
                        masked_value = f"{str(value)[:8]}..." if len(str(value)) > 8 else "TOO_SHORT"
                        self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG]   {key}: {masked_value}")
                    else:
                        self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG]   {key}: {value}")
            
            connector = exchange_class(**init_params)
            
            # 🔍 DEBUG: 检查连接器创建后的状态（主方法）
            if connector_name == "bybit":
                self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] Connector created successfully")
                
                # 检查是否有必要的属性
                attrs_to_check = ['_web_assistants_factory', '_auth', '_throttler', '_time_synchronizer', '_order_book_tracker', '_user_stream_tracker', '_orderbook_ds']
                for attr in attrs_to_check:
                    has_attr = hasattr(connector, attr)
                    attr_value = getattr(connector, attr, None) if has_attr else None
                    self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] {attr}: {'EXISTS' if has_attr else 'MISSING'} (value: {'SET' if attr_value else 'NONE'})")
                    
                    # 对 order_book_tracker 进行更详细的检查
                    if attr == '_order_book_tracker' and has_attr and attr_value:
                        try:
                            order_books_count = len(attr_value.order_books) if hasattr(attr_value, 'order_books') else 0
                            self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG]   -> order_book_tracker has {order_books_count} order books")
                            
                            # 检查是否在运行
                            is_running = hasattr(attr_value, '_order_book_stream_listener_task') and attr_value._order_book_stream_listener_task is not None
                            self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG]   -> order_book_tracker running: {is_running}")
                        except Exception as e:
                            self._logger.error(f"❌ [BYBIT MAIN INIT DEBUG]   -> order_book_tracker inspection error: {e}")
                
                # 检查连接器类型和继承
                connector_type = type(connector).__name__
                connector_mro = [cls.__name__ for cls in type(connector).__mro__]
                self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] Connector type: {connector_type}")
                self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] Connector MRO: {connector_mro}")
                
                # 检查连接器的初始状态
                try:
                    status_before_network = connector.status_dict if hasattr(connector, 'status_dict') else "NO_STATUS_DICT"
                    self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] Status before network start: {status_before_network}")
                except Exception as status_error:
                    self._logger.error(f"❌ [BYBIT MAIN INIT DEBUG] Failed to get status: {status_error}")
                
                # 检查重要的内部属性
                internal_attrs = ['_trading_pairs', '_trading_rules', '_account_balances']
                for attr in internal_attrs:
                    has_attr = hasattr(connector, attr)
                    if has_attr:
                        attr_value = getattr(connector, attr, None)
                        if isinstance(attr_value, dict):
                            self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] {attr}: dict with {len(attr_value)} items")
                        elif isinstance(attr_value, list):
                            self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] {attr}: list with {len(attr_value)} items")
                        else:
                            self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] {attr}: {type(attr_value).__name__}")
                    else:
                        self._logger.info(f"🔍 [BYBIT MAIN INIT DEBUG] {attr}: MISSING")
            
            # 等待连接器初始化完成
            try:
                self._logger.info(f"🔧 [BYBIT NETWORK DEBUG] Starting network for {connector_name}...")
                await connector.start_network()
                self._logger.info(f"✅ [BYBIT NETWORK DEBUG] Network started successfully")
                
                # 网络启动后再次检查关键属性
                if connector_name == "bybit":
                    self._logger.info(f"🔍 [BYBIT NETWORK DEBUG] Post-network startup check...")
                    
                    # 检查网络启动后的状态
                    try:
                        status_after_network = connector.status_dict
                        self._logger.info(f"🔍 [BYBIT NETWORK DEBUG] Status after network start: {status_after_network}")
                    except Exception as status_error:
                        self._logger.error(f"❌ [BYBIT NETWORK DEBUG] Failed to get status after network start: {status_error}")
                    
                    # 检查关键组件是否被网络启动过程初始化
                    network_attrs = ['_web_assistants_factory', '_order_book_tracker', '_user_stream_tracker']
                    for attr in network_attrs:
                        has_attr = hasattr(connector, attr)
                        attr_value = getattr(connector, attr, None) if has_attr else None
                        self._logger.info(f"🔍 [BYBIT NETWORK DEBUG] After network start - {attr}: {'EXISTS' if has_attr else 'MISSING'} (value: {'SET' if attr_value else 'NONE'})")
                    
                    # 记录 web assistants factory 状态
                    if hasattr(connector, '_web_assistants_factory') and connector._web_assistants_factory:
                        self._logger.info(f"✅ [BYBIT NETWORK DEBUG] Web assistants factory exists and is set")
                    else:
                        self._logger.error(f"❌ [BYBIT NETWORK DEBUG] Web assistants factory missing after network start!")
                    
                    # 如果仍然缺少订单簿跟踪器，尝试手动创建
                    if not hasattr(connector, '_order_book_tracker') or not connector._order_book_tracker:
                        self._logger.error(f"❌ [BYBIT NETWORK DEBUG] Order book tracker still missing!")
                        try:
                            self._logger.info(f"🔧 [BYBIT NETWORK DEBUG] Attempting to manually initialize order book tracker...")
                            
                            if hasattr(connector, '_create_order_book_data_source') and hasattr(connector, '_set_order_book_tracker'):
                                from hummingbot.core.data_type.order_book_tracker import OrderBookTracker
                                
                                # 创建订单簿数据源
                                orderbook_ds = connector._create_order_book_data_source()
                                self._logger.info(f"🔧 [BYBIT NETWORK DEBUG] Created order book data source")
                                
                                # 创建并设置订单簿跟踪器
                                tracker = OrderBookTracker(data_source=orderbook_ds, trading_pairs=connector._trading_pairs)
                                connector._set_order_book_tracker(tracker)
                                self._logger.info(f"✅ [BYBIT NETWORK DEBUG] Manually created order book tracker")
                                
                                # 启动订单簿跟踪器
                                if hasattr(connector, '_order_book_tracker') and connector._order_book_tracker:
                                    connector._order_book_tracker.start()
                                    self._logger.info(f"✅ [BYBIT NETWORK DEBUG] Order book tracker started")
                                else:
                                    self._logger.error(f"❌ [BYBIT NETWORK DEBUG] Order book tracker still not set after creation")
                                
                            else:
                                self._logger.error(f"❌ [BYBIT NETWORK DEBUG] Missing required methods for order book tracker creation")
                                
                        except Exception as tracker_error:
                            self._logger.error(f"❌ [BYBIT NETWORK DEBUG] Failed to manually create order book tracker: {tracker_error}")
                            import traceback
                            self._logger.error(f"❌ [BYBIT NETWORK DEBUG] Tracker error traceback: {traceback.format_exc()}")
                    
                    # 检查是否有网络迭代器
                    if hasattr(connector, '_network_iterator'):
                        network_status = "running" if connector._network_iterator else "not_running"
                        self._logger.info(f"🔍 [BYBIT NETWORK DEBUG] Network iterator: {network_status}")
                    else:
                        self._logger.warning(f"⚠️ [BYBIT NETWORK DEBUG] No network iterator found")
                
                # 🔍 DEBUG: 为特定连接器手动触发余额更新
                if connector_name == "hyperliquid":
                    self._logger.info(f"🔍 [HYPERLIQUID DEBUG] Manually triggering balance update for {connector_name} (fallback method)")
                    try:
                        if hasattr(connector, '_update_balances'):
                            await connector._update_balances()
                            self._logger.info(f"✅ [HYPERLIQUID DEBUG] Manual balance update completed (fallback)")
                        else:
                            self._logger.warning(f"⚠️ [HYPERLIQUID DEBUG] No _update_balances method found (fallback)")
                    except Exception as e:
                        self._logger.error(f"❌ [HYPERLIQUID DEBUG] Manual balance update failed (fallback): {e}")
                
                elif connector_name == "bybit":
                    self._logger.info(f"🔍 [BYBIT DEBUG] Manually triggering initialization for {connector_name}")
                    try:
                        # 检查 Bybit 连接器的网络组件
                        if hasattr(connector, '_user_stream_tracker'):
                            user_stream_status = "initialized" if connector._user_stream_tracker else "not_initialized"
                            self._logger.info(f"🔍 [BYBIT DEBUG] User stream tracker: {user_stream_status}")
                        
                        if hasattr(connector, '_web_assistants_factory'):
                            api_factory_status = "initialized" if connector._web_assistants_factory else "not_initialized"
                            self._logger.info(f"🔍 [BYBIT DEBUG] Web assistants factory: {api_factory_status}")
                        
                        # 检查订单簿跟踪器状态
                        if hasattr(connector, '_order_book_tracker'):
                            ob_tracker = connector._order_book_tracker
                            if ob_tracker:
                                ob_count = len(ob_tracker.order_books)
                                self._logger.info(f"🔍 [BYBIT DEBUG] Order book tracker has {ob_count} order books")
                                
                                # 检查是否有 BTC-USDT 的订单簿
                                if trading_pair in ob_tracker.order_books:
                                    order_book = ob_tracker.order_books[trading_pair]
                                    bid_count = len(order_book.bid_entries())
                                    ask_count = len(order_book.ask_entries())
                                    self._logger.info(f"🔍 [BYBIT DEBUG] {trading_pair} order book: {bid_count} bids, {ask_count} asks")
                                else:
                                    self._logger.warning(f"⚠️ [BYBIT DEBUG] No order book found for {trading_pair}")
                                
                                # 尝试手动启动订单簿数据流
                                try:
                                    self._logger.info(f"🔧 [BYBIT DEBUG] Attempting to manually start order book tracker...")
                                    if hasattr(ob_tracker, '_order_book_stream_listener_task') and not ob_tracker._order_book_stream_listener_task:
                                        ob_tracker.start()
                                        self._logger.info(f"✅ [BYBIT DEBUG] Order book tracker started")
                                    else:
                                        self._logger.info(f"🔍 [BYBIT DEBUG] Order book tracker already running")
                                        
                                except Exception as ob_error:
                                    self._logger.error(f"❌ [BYBIT DEBUG] Failed to start order book tracker: {ob_error}")
                            else:
                                self._logger.error(f"❌ [BYBIT DEBUG] Order book tracker is None")
                        else:
                            self._logger.error(f"❌ [BYBIT DEBUG] No order book tracker found")
                        
                        # 尝试手动调用余额更新
                        if hasattr(connector, '_update_balances'):
                            await connector._update_balances()
                            self._logger.info(f"✅ [BYBIT DEBUG] Manual balance update completed")
                            
                            # 检查余额是否真的更新了
                            balance_count = len(connector._account_balances)
                            self._logger.info(f"🔍 [BYBIT DEBUG] Balance count after update: {balance_count}")
                        else:
                            self._logger.warning(f"⚠️ [BYBIT DEBUG] No _update_balances method found")
                            
                        # 等待一下让订单簿数据加载
                        self._logger.info(f"🔧 [BYBIT DEBUG] Waiting 5 seconds for order book data to load...")
                        await asyncio.sleep(5)
                        
                        # 再次检查订单簿
                        if hasattr(connector, '_order_book_tracker') and connector._order_book_tracker:
                            ob_tracker = connector._order_book_tracker
                            ob_count = len(ob_tracker.order_books)
                            self._logger.info(f"🔍 [BYBIT DEBUG] After waiting, order book tracker has {ob_count} order books")
                            
                            if trading_pair in ob_tracker.order_books:
                                order_book = ob_tracker.order_books[trading_pair]
                                bid_count = len(order_book.bid_entries())
                                ask_count = len(order_book.ask_entries())
                                self._logger.info(f"✅ [BYBIT DEBUG] {trading_pair} order book now has: {bid_count} bids, {ask_count} asks")
                            else:
                                self._logger.warning(f"⚠️ [BYBIT DEBUG] Still no order book found for {trading_pair}")
                            
                    except Exception as e:
                        self._logger.error(f"❌ [BYBIT DEBUG] Manual initialization failed: {e}")
                        import traceback
                        self._logger.error(f"❌ [BYBIT DEBUG] Stack trace: {traceback.format_exc()}")
                
                # 等待连接器完全初始化 - 增加更多等待时间和检查项目
                max_wait_cycles = 15  # 最多等待30秒
                for i in range(max_wait_cycles):
                    status_dict = connector.status_dict
                    
                    # 检查关键组件是否已就绪
                    account_balance_ready = status_dict.get('account_balance', False)
                    symbols_mapping_ready = status_dict.get('symbols_mapping_initialized', False) 
                    trading_rules_ready = status_dict.get('trading_rule_initialized', False)
                    
                    # 对于订单簿，我们先检查是否至少有一个交易对的订单簿数据
                    order_books_ready = status_dict.get('order_books_initialized', False)
                    if not order_books_ready and hasattr(connector, '_order_book_tracker'):
                        # 检查是否至少有一个订单簿有数据
                        try:
                            for trading_pair in connector._trading_pairs or [trading_pair]:
                                order_book = connector._order_book_tracker.order_books.get(trading_pair)
                                if order_book and len(order_book.bid_entries()) > 0 and len(order_book.ask_entries()) > 0:
                                    order_books_ready = True
                                    self._logger.info(f"Order book data found for {trading_pair}")
                                    break
                        except Exception as e:
                            self._logger.debug(f"Error checking order book data: {e}")
                    
                    # 记录当前状态
                    self._logger.info(f"Connector {connector_name} status (attempt {i+1}/{max_wait_cycles}): "
                                    f"balance={account_balance_ready}, symbols={symbols_mapping_ready}, "
                                    f"rules={trading_rules_ready}, orderbooks={order_books_ready}")
                    
                    # 如果关键组件都准备好了，就继续
                    if account_balance_ready and symbols_mapping_ready and trading_rules_ready:
                        self._logger.info(f"Core components ready for {connector_name}")
                        break
                    
                    await asyncio.sleep(2)
                
                # 最终状态检查
                final_status = connector.status_dict
                if not final_status.get('account_balance', False):
                    # 对于 Bybit，如果网络有问题，我们放宽余额要求
                    if connector_name == "bybit":
                        self._logger.warning(f"⚠️ [BYBIT DEBUG] Account balance not loaded, but continuing anyway due to network issues")
                    else:
                        raise RuntimeError(f"Account balance not loaded for {connector_name}. Cannot initialize connector without account information.")
                
                # 对于其他组件，只记录警告但不阻塞
                if not connector.ready:
                    not_ready_components = [k for k, v in final_status.items() if not v]
                    self._logger.warning(f"Connector {connector_name} not fully ready, but continuing. Missing components: {not_ready_components}")
                
                self._logger.info(f"Connector {connector_name} initialized successfully. Final status: {final_status}")
                
            except Exception as e:
                self._logger.error(f"Failed to initialize connector {connector_name}: {e}")
                raise RuntimeError(f"Connector {connector_name} failed to initialize: {e}")
            
            # 缓存连接器（覆盖可能失效的预初始化连接器）
            self._connectors[connector_name] = connector
            
            return connector

    async def place_order(
        self,
        connector_name: str,
        trading_pair: str,
        amount: float,
        is_buy: bool,
        order_type: str = "MARKET",
        price: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        在指定的CEX连接器上执行实际下单操作
        
        参数:
            connector_name (str): 交易所名称，如 "binance"、"hyperliquid" 等
            trading_pair (str): 交易对，如 "BTC-USDT"
            amount (float): 下单数量（基础资产数量）
            is_buy (bool): True=买入，False=卖出  
            order_type (str): 订单类型，"MARKET"/"LIMIT"/"LIMIT_MAKER"
            price (float): 订单价格（限价单必须提供）
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
                "timestamp": float,
                "error": str (如果失败)
            }
        """
        try:
            # 参数验证
            if not validate_exchange_support(connector_name):
                return {
                    "success": False,
                    "error": f"Unsupported exchange: {connector_name}"
                }
                
            if amount <= 0:
                return {
                    "success": False,
                    "error": "Amount must be greater than 0"
                }
                
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
            
            # 获取或创建连接器实例
            try:
                connector = await self._get_or_create_connector(connector_name, trading_pair, wait_for_orderbook=False)
                
                # 🔍 DEBUG: 专门为 Bybit 添加 API 认证检查
                if connector_name == "bybit":
                    self._logger.info(f"🔍 [BYBIT API DEBUG] Checking API configuration...")
                    
                    # 检查 API 密钥配置
                    if hasattr(connector, '_web_assistants_factory') and connector._web_assistants_factory:
                        self._logger.info(f"✅ [BYBIT API DEBUG] Web assistants factory found")
                        if hasattr(connector, '_auth') and connector._auth:
                            auth = connector._auth
                            api_key = getattr(auth, '_api_key', 'NOT_SET')
                            api_secret = getattr(auth, '_secret_key', 'NOT_SET')
                            
                            # 只显示前几个字符，保护隐私
                            masked_key = f"{api_key[:8]}..." if len(api_key) > 8 else "TOO_SHORT"
                            masked_secret = f"{api_secret[:8]}..." if len(api_secret) > 8 else "TOO_SHORT"
                            
                            self._logger.info(f"🔍 [BYBIT API DEBUG] API Key: {masked_key}")
                            self._logger.info(f"🔍 [BYBIT API DEBUG] API Secret: {masked_secret}")
                        else:
                            self._logger.error(f"❌ [BYBIT API DEBUG] No authentication found in connector!")
                    else:
                        self._logger.error(f"❌ [BYBIT API DEBUG] No web assistants factory found!")
                    
                    # 检查账户余额状态
                    balance_count = len(connector._account_balances)
                    self._logger.info(f"🔍 [BYBIT API DEBUG] Current balance count: {balance_count}")
                    
                    if balance_count > 0:
                        sample_balances = dict(list(connector._account_balances.items())[:3])
                        self._logger.info(f"🔍 [BYBIT API DEBUG] Sample balances: {sample_balances}")
                    
                    # 检查连接器状态
                    status_dict = connector.status_dict
                    self._logger.info(f"🔍 [BYBIT API DEBUG] Connector status: {status_dict}")
                    
                    # 检查订单簿状态
                    if hasattr(connector, '_order_book_tracker') and connector._order_book_tracker:
                        ob_tracker = connector._order_book_tracker
                        ob_count = len(ob_tracker.order_books)
                        self._logger.info(f"🔍 [BYBIT API DEBUG] Order book tracker has {ob_count} order books")
                        
                        if trading_pair in ob_tracker.order_books:
                            order_book = ob_tracker.order_books[trading_pair]
                            bid_count = len(order_book.bid_entries())
                            ask_count = len(order_book.ask_entries())
                            self._logger.info(f"🔍 [BYBIT API DEBUG] {trading_pair} order book: {bid_count} bids, {ask_count} asks")
                            
                            if bid_count == 0 or ask_count == 0:
                                self._logger.error(f"❌ [BYBIT API DEBUG] Order book for {trading_pair} has insufficient data!")
                                # 尝试快速重新获取订单簿数据
                                try:
                                    self._logger.info(f"🔧 [BYBIT API DEBUG] Attempting to force refresh order book data...")
                                    await asyncio.sleep(3)  # 等待3秒让数据加载
                                    
                                    # 再次检查
                                    bid_count_after = len(order_book.bid_entries())
                                    ask_count_after = len(order_book.ask_entries())
                                    self._logger.info(f"🔍 [BYBIT API DEBUG] After refresh - {trading_pair}: {bid_count_after} bids, {ask_count_after} asks")
                                except Exception as refresh_error:
                                    self._logger.error(f"❌ [BYBIT API DEBUG] Order book refresh failed: {refresh_error}")
                        else:
                            self._logger.error(f"❌ [BYBIT API DEBUG] No order book found for {trading_pair}!")
                            return {
                                "success": False,
                                "error": f"Order book not available for {trading_pair}. Please wait for market data to load.",
                                "exchange": connector_name,
                                "trading_pair": trading_pair,
                                "amount": amount,
                                "side": "BUY" if is_buy else "SELL"
                            }
                    
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
                price = None
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
            
            # 下单前检查余额（仅限价单检查，市价单让交易所验证）
            try:
                base_asset, quote_asset = trading_pair.split("-")
                
                if is_buy and order_type != "MARKET":
                    # 限价买入时检查 quote 资产余额
                    quote_balance = connector._account_balances.get(quote_asset, Decimal("0"))
                    estimated_cost = Decimal(str(amount)) * Decimal(str(price))
                    
                    if quote_balance < estimated_cost:
                        return {
                            "success": False,
                            "error": f"Insufficient {quote_asset} balance. Available: {quote_balance}, Required: {estimated_cost}",
                            "exchange": connector_name,
                            "trading_pair": trading_pair,
                            "amount": amount,
                            "side": "BUY",
                            "available_balance": float(quote_balance),
                            "required_amount": float(estimated_cost)
                        }
                elif not is_buy:
                    # 卖出时检查 base 资产余额
                    base_balance = connector._account_balances.get(base_asset, Decimal("0"))
                    required_amount = Decimal(str(amount))
                    
                    if base_balance < required_amount:
                        return {
                            "success": False,
                            "error": f"Insufficient {base_asset} balance. Available: {base_balance}, Required: {required_amount}",
                            "exchange": connector_name,
                            "trading_pair": trading_pair,
                            "amount": amount,
                            "side": "SELL",
                            "available_balance": float(base_balance),
                            "required_amount": float(required_amount)
                        }
                
                if order_type == "MARKET" and is_buy:
                    self._logger.info(f"Skipping balance check for market buy order on {trading_pair}")
                else:
                    self._logger.info(f"Balance check passed for {trading_pair} {order_type} {'BUY' if is_buy else 'SELL'} order")
                
            except Exception as balance_check_error:
                self._logger.warning(f"Error during balance check: {balance_check_error}. Proceeding with order...")
            
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
                
                self._logger.info(f"Order submitted to {connector_name} with local ID: {order_id}")
                
                # 立即返回成功结果，不等待订单状态
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
                self._logger.error(f"Error during order execution on {connector_name}: {order_error}")
                return {
                    "success": False,
                    "error": f"Order execution failed: {str(order_error)}",
                    "exchange": connector_name,
                    "trading_pair": trading_pair,
                    "amount": amount,
                    "side": "BUY" if is_buy else "SELL"
                }
            
        except Exception as e:
            self._logger.error(f"Error placing order on {connector_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "exchange": connector_name,
                "trading_pair": trading_pair,
                "amount": amount,
                "side": "BUY" if is_buy else "SELL"
            }

    async def get_order_status(
        self,
        connector_name: str,
        order_id: str,
        trading_pair: str,
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """
        查询订单状态（带重试机制）
        
        参数:
            connector_name (str): 交易所名称
            order_id (str): 订单ID
            trading_pair (str): 交易对
            max_retries (int): 最大重试次数，默认5次
            
        返回:
            dict: 订单状态信息
            {
                "success": bool,
                "order_id": str,
                "exchange": str,
                "trading_pair": str,
                "status": str,
                "filled_amount": float,
                "remaining_amount": float,
                "average_price": float,
                "retry_attempts": int,
                "error": str (如果失败)
            }
        """
        try:
            if not validate_exchange_support(connector_name):
                return {
                    "success": False,
                    "error": f"Unsupported exchange: {connector_name}",
                    "order_id": order_id
                }
            
            # 获取连接器实例
            try:
                connector = await self._get_or_create_connector(connector_name, trading_pair, wait_for_orderbook=False)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to get connector for {connector_name}: {str(e)}",
                    "order_id": order_id,
                    "exchange": connector_name
                }
            
            # 多次重试检查订单状态
            retry_count = 0
            order_found = False
            order = None
            
            while retry_count < max_retries:
                # 等待订单状态更新
                wait_time = 2 + retry_count  # 递增等待时间：2s, 3s, 4s, 5s, 6s
                await asyncio.sleep(wait_time)
                
                # 检查订单是否存在于连接器的订单跟踪器中
                order_tracker = connector._order_tracker
                if hasattr(order_tracker, 'all_orders') and order_id in order_tracker.all_orders:
                    order = order_tracker.all_orders[order_id]
                    order_found = True
                    
                    if hasattr(order, 'current_state'):
                        order_state = str(order.current_state)
                        self._logger.info(f"Order {order_id} found (attempt {retry_count + 1}): state = {order_state}")
                        
                        # 如果订单状态明确（成功或失败），停止重试
                        if any(state in order_state.upper() for state in ['OPEN', 'PARTIALLY_FILLED', 'FILLED', 'COMPLETED', 'FAILED', 'CANCELLED', 'REJECTED']):
                            break
                        
                        # 如果是其他状态（如 PENDING），继续重试
                        self._logger.info(f"Order {order_id} in intermediate state: {order_state}, retrying...")
                    break
                else:
                    self._logger.warning(f"Order {order_id} not found in tracker (attempt {retry_count + 1}/{max_retries})")
                
                retry_count += 1
            
            if order_found and order:
                order_state = str(order.current_state) if hasattr(order, 'current_state') else 'UNKNOWN'
                
                result = {
                    "success": True,
                    "order_id": order_id,
                    "exchange": connector_name,
                    "trading_pair": trading_pair,
                    "status": order_state,
                    "filled_amount": float(order.executed_amount_base) if hasattr(order, 'executed_amount_base') else 0,
                    "remaining_amount": float(order.amount - order.executed_amount_base) if hasattr(order, 'amount') and hasattr(order, 'executed_amount_base') else 0,
                    "average_price": float(order.average_executed_price) if hasattr(order, 'average_executed_price') and order.average_executed_price else None,
                    "created_timestamp": getattr(order, 'creation_timestamp', None),
                    "last_update_timestamp": getattr(order, 'last_update_timestamp', None),
                    "exchange_order_id": getattr(order, 'exchange_order_id', None),
                    "retry_attempts": retry_count
                }
                
                # 如果订单失败，添加详细的失败信息
                if any(fail_state in order_state.upper() for fail_state in ['FAILED', 'CANCELLED', 'REJECTED']):
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
                    "error": f"Order {order_id} not found after {max_retries} attempts",
                    "order_id": order_id,
                    "exchange": connector_name,
                    "retry_attempts": max_retries
                }
                
        except Exception as e:
            self._logger.error(f"Error getting order status from {connector_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "order_id": order_id,
                "exchange": connector_name
            }

    async def cancel_order(
        self,
        connector_name: str,
        order_id: str,
        trading_pair: str
    ) -> Dict[str, Any]:
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
            if not validate_exchange_support(connector_name):
                return {
                    "success": False,
                    "error": f"Unsupported exchange: {connector_name}",
                    "order_id": order_id
                }
            
            # 获取连接器实例
            connector = await self._get_or_create_connector(connector_name, trading_pair, wait_for_orderbook=False)
            
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
            self._logger.error(f"Error canceling order on {connector_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "order_id": order_id,
                "exchange": connector_name
            }

    def get_supported_exchanges(self) -> list:
        """
        获取支持的交易所列表
        
        返回:
            list: 支持的交易所名称列表
        """
        return get_supported_exchanges()

    def get_exchange_info(self, connector_name: str) -> dict:
        """
        获取交易所信息
        
        参数:
            connector_name (str): 交易所名称
            
        返回:
            dict: 交易所配置信息
        """
        if not validate_exchange_support(connector_name):
            return {
                "error": f"Unsupported exchange: {connector_name}"
            }
        
        config = get_exchange_config(connector_name)
        return {
            "name": connector_name,
            "exchange_class": config["exchange_class"].__name__,
            "data_source_class": config["data_source_class"].__name__,
            "required_params": list(config["required_params"].keys())
        }


# 全局实例
_order_manager_instance = None

def get_order_manager() -> OrderManager:
    """获取全局订单管理器实例"""
    global _order_manager_instance
    if _order_manager_instance is None:
        _order_manager_instance = OrderManager(start_background_initialization=False)
    
    return _order_manager_instance


async def main():
    """测试用的主函数"""
    print("FluxLayer Order Manager - Test Mode")
    
    order_manager = get_order_manager()
    
    # 显示支持的交易所
    supported_exchanges = order_manager.get_supported_exchanges()
    print(f"Supported exchanges: {supported_exchanges}")
    
    # 测试下单功能
    print("\n=== Testing Order Placement ===")

    connector_name = "okx"
    trading_pair = "BTC-USDT"
    amount = 0.0000546
    # amount = 0.00086


    # 示例：市价买单
    result = await order_manager.place_order(
        connector_name=connector_name,
        trading_pair=trading_pair,
        amount=amount,
        is_buy=True,
        order_type="MARKET"
    )
    
    print(f"Order placement result: {result}")
    
    # 如果下单成功，测试查询订单状态
    if result.get("success") and result.get("order_id"):
        print(f"\n=== Testing Order Status Query ===")
        
        order_id = result["order_id"]
        exchange = result["exchange"]
        trading_pair = result["trading_pair"]
        
        # 查询订单状态
        status_result = await order_manager.get_order_status(
            connector_name=exchange,
            order_id=order_id,
            trading_pair=trading_pair,
            max_retries=3
        )
        
        print(f"Order status result: {status_result}")


if __name__ == "__main__":
    # 运行测试
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()