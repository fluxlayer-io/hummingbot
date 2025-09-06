"""
FluxLayer Exchange Constants

存放所有支持的交易所配置和常量
"""
import os

from hummingbot.connector.exchange.bybit import bybit_constants as BYBIT_CONSTANTS
from hummingbot.connector.exchange.bybit.bybit_api_order_book_data_source import BybitAPIOrderBookDataSource
from hummingbot.connector.exchange.bybit.bybit_exchange import BybitExchange

from hummingbot.connector.exchange.okx import okx_constants as OKX_CONSTANTS
from hummingbot.connector.exchange.okx.okx_api_order_book_data_source import OkxAPIOrderBookDataSource
from hummingbot.connector.exchange.okx.okx_exchange import OkxExchange

from hummingbot.connector.exchange.hyperliquid import hyperliquid_constants as HYPERLIQUID_CONSTANTS
from hummingbot.connector.exchange.hyperliquid.hyperliquid_api_order_book_data_source import HyperliquidAPIOrderBookDataSource
from hummingbot.connector.exchange.hyperliquid.hyperliquid_exchange import HyperliquidExchange

from hummingbot.connector.exchange.binance import binance_constants as BINANCE_CONSTANTS
from hummingbot.connector.exchange.binance.binance_api_order_book_data_source import BinanceAPIOrderBookDataSource
from hummingbot.connector.exchange.binance.binance_exchange import BinanceExchange


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

# 稳定币定义 - 所有主流稳定币都按 1:1 美元处理
STABLECOINS = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDD', 'USDP', 'FRAX', 'LUSD', 'sUSD', 'GUSD'}

# Hyperliquid 符号映射
HYPERLIQUID_SYMBOL_MAPPING = {
    "BTC": "UBTC",
    "USDC": "USDT0"
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


def get_trading_symbol(exchange_name: str, base_token: str, quote_token: str = "USDT") -> str:
    """
    获取交易所特定的交易对符号，优化以支持新的RFQ逻辑
    
    参数:
        exchange_name (str): 交易所名称
        base_token (str): 基础代币
        quote_token (str): 报价代币，默认为 USDT
        
    返回:
        str: 格式化的交易对符号
        
    抛出:
        ValueError: 如果尝试生成无效的交易对（如稳定币对稳定币）
    """
    print(f"🔀 [SYMBOL DEBUG] get_trading_symbol called: exchange={exchange_name}, base={base_token}, quote={quote_token}")
    
    # 检查是否为无效的稳定币对稳定币交易对
    if is_stablecoin(base_token) and is_stablecoin(quote_token):
        error_msg = f"Invalid trading pair: {base_token}-{quote_token} (stablecoin to stablecoin)"
        print(f"❌ [SYMBOL DEBUG] {error_msg}")
        raise ValueError(error_msg)
    
    if exchange_name == "hyperliquid":
        # Hyperliquid 特殊映射逻辑
        if base_token == "BTC":
            # BTC -> UBTC-USDC
            mapped_base = "UBTC"
            mapped_quote = "USDC"
            print(f"🔀 [SYMBOL DEBUG] Hyperliquid: BTC mapped to {mapped_base}-{mapped_quote}")
            result = f"{mapped_base}-{mapped_quote}"
        elif base_token == "ETH":
            # ETH -> UETH-USDC  
            mapped_base = "UETH"
            mapped_quote = "USDC"
            print(f"🔀 [SYMBOL DEBUG] Hyperliquid: ETH mapped to {mapped_base}-{mapped_quote}")
            result = f"{mapped_base}-{mapped_quote}"
        else:
            # 其他代币使用 X-USDC 格式
            mapped_base = base_token
            mapped_quote = "USDC"
            print(f"🔀 [SYMBOL DEBUG] Hyperliquid: {base_token} using as-is with USDC quote")
            result = f"{mapped_base}-{mapped_quote}"
            
        print(f"✅ [SYMBOL DEBUG] Hyperliquid result: {result}")
        return result
    else:
        # 其他交易所使用标准格式
        # 优先使用主流代币对稳定币的配对 (如 BTC-USDT, ETH-USDT)
        if is_stablecoin(base_token) and not is_stablecoin(quote_token):
            # 如果base是稳定币但quote不是，这可能不是常见的交易对
            print(f"⚠️ [SYMBOL DEBUG] Unusual pair: stablecoin {base_token} as base with {quote_token} as quote")
        
        result = f"{base_token}-{quote_token}"
        print(f"✅ [SYMBOL DEBUG] {exchange_name} result: {result}")
        return result


def get_hyperliquid_supported_tokens() -> list:
    """
    获取 Hyperliquid 支持的基础代币列表
    
    返回:
        list: 支持的基础代币列表
    """
    return ["BTC", "ETH"]  # 基于API调查结果，支持 BTC->UBTC-USDC 和 ETH->UETH-USDC


def validate_hyperliquid_token(base_token: str) -> bool:
    """
    验证代币是否在 Hyperliquid 上受支持
    
    参数:
        base_token (str): 基础代币符号
        
    返回:
        bool: 是否受支持
    """
    return base_token in get_hyperliquid_supported_tokens()


def is_stablecoin(token: str) -> bool:
    """
    检查代币是否为稳定币
    
    参数:
        token (str): 代币符号
        
    返回:
        bool: 是否为稳定币
    """
    return token.upper() in STABLECOINS


def get_stablecoin_price() -> float:
    """
    获取稳定币价格 - 始终返回 1.0
    
    返回:
        float: 稳定币价格 (1.0)
    """
    return 1.0


# 自动加载环境变量中的API密钥
load_api_keys_from_env()