"""
FluxLayer Exchange Constants

存放所有支持的交易所配置和常量
"""
import os

# 导入交易所相关模块
from hummingbot.connector.exchange.binance import binance_constants as BINANCE_CONSTANTS
from hummingbot.connector.exchange.binance.binance_api_order_book_data_source import BinanceAPIOrderBookDataSource
from hummingbot.connector.exchange.binance.binance_exchange import BinanceExchange

from hummingbot.connector.exchange.bybit import bybit_constants as BYBIT_CONSTANTS
from hummingbot.connector.exchange.bybit.bybit_api_order_book_data_source import BybitAPIOrderBookDataSource
from hummingbot.connector.exchange.bybit.bybit_exchange import BybitExchange

from hummingbot.connector.exchange.okx import okx_constants as OKX_CONSTANTS
from hummingbot.connector.exchange.okx.okx_api_order_book_data_source import OkxAPIOrderBookDataSource
from hummingbot.connector.exchange.okx.okx_exchange import OkxExchange

from hummingbot.connector.exchange.hyperliquid import hyperliquid_constants as HYPERLIQUID_CONSTANTS
from hummingbot.connector.exchange.hyperliquid.hyperliquid_api_order_book_data_source import HyperliquidAPIOrderBookDataSource
from hummingbot.connector.exchange.hyperliquid.hyperliquid_exchange import HyperliquidExchange


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
    "bybit": {
        "exchange_class": BybitExchange,
        "data_source_class": BybitAPIOrderBookDataSource,
        "constants": BYBIT_CONSTANTS,
        "required_params": {
            "bybit_api_key": "",
            "bybit_api_secret": ""
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

# 链上Gas代币映射
CHAIN_GAS_TOKEN_MAP = {
    "Ethereum": "ETH",
    "BSC": "BNB", 
    "Polygon": "MATIC",
    "Arbitrum": "ETH",
    "BTC": "BTC",
    "Solana": "SOL",
}


def load_api_keys_from_env():
    """从环境变量加载API密钥到交易所配置中"""
    try:
        # 更新 Binance API 密钥
        EXCHANGES["binance"]["required_params"]["binance_api_key"] = os.getenv("BINANCE_API_KEY", "")
        EXCHANGES["binance"]["required_params"]["binance_api_secret"] = os.getenv("BINANCE_API_SECRET", "")
        
        # 更新 Bybit API 密钥
        EXCHANGES["bybit"]["required_params"]["bybit_api_key"] = os.getenv("BYBIT_API_KEY", "")
        EXCHANGES["bybit"]["required_params"]["bybit_api_secret"] = os.getenv("BYBIT_API_SECRET", "")
        
        # 更新 OKX API 密钥
        EXCHANGES["okx"]["required_params"]["okx_api_key"] = os.getenv("OKX_API_KEY", "")
        EXCHANGES["okx"]["required_params"]["okx_secret_key"] = os.getenv("OKX_SECRET_KEY", "")
        EXCHANGES["okx"]["required_params"]["okx_passphrase"] = os.getenv("OKX_PASSPHRASE", "")
        
        # 更新 Hyperliquid API 密钥
        EXCHANGES["hyperliquid"]["required_params"]["hyperliquid_api_key"] = os.getenv("HYPERLIQUID_API_KEY", "")
        EXCHANGES["hyperliquid"]["required_params"]["hyperliquid_api_secret"] = os.getenv("HYPERLIQUID_API_SECRET", "")
        EXCHANGES["hyperliquid"]["required_params"]["use_vault"] = os.getenv("HYPERLIQUID_USE_VAULT", "False").lower() == "true"
        
    except Exception as e:
        print(f"Warning: Error loading API keys from environment: {e}")


def get_supported_exchanges():
    """获取支持的交易所列表"""
    return list(EXCHANGES.keys())


def get_exchange_config(exchange_name: str):
    """
    获取指定交易所的配置
    
    参数:
        exchange_name (str): 交易所名称
        
    返回:
        dict: 交易所配置，如果不存在则返回 None
    """
    return EXCHANGES.get(exchange_name)


def validate_exchange_support(exchange_name: str) -> bool:
    """
    验证交易所是否受支持
    
    参数:
        exchange_name (str): 交易所名称
        
    返回:
        bool: True 如果支持，False 如果不支持
    """
    return exchange_name in EXCHANGES


# 自动加载环境变量中的API密钥
load_api_keys_from_env()