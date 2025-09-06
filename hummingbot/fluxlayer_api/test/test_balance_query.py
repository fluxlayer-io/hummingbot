#!/usr/bin/env python3
"""
余额查询测试文件
测试从 exchange_constants.py 中所有支持的交易所获取余额信息
"""

import asyncio
import os
import sys
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.append(project_root)

from hummingbot.fluxlayer_api.exchange_constants import EXCHANGES, load_api_keys_from_env
from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler


class BalanceQueryService:
    """余额查询服务类"""
    
    def __init__(self):
        self.initialized_exchanges = {}
        self.client_config_adapter = None
        self._setup_client_config()
    
    def _setup_client_config(self):
        """设置客户端配置"""
        try:
            client_config = ClientConfigMap()
            self.client_config_adapter = ClientConfigAdapter(client_config)
        except Exception as e:
            print(f"⚠️ Warning: Failed to setup client config: {e}")
            self.client_config_adapter = None
    
    async def get_exchange_balances(self, exchange_name: str) -> Dict[str, Any]:
        """
        获取指定交易所的余额信息
        
        Args:
            exchange_name: 交易所名称 (binance, bybit, okx, hyperliquid)
            
        Returns:
            {
                "exchange": "binance",
                "balances": {
                    "BTC": {"available": 0.001, "total": 0.001},
                    "USDT": {"available": 100.0, "total": 100.0}
                },
                "timestamp": 1625000000,
                "status": "success"|"error",
                "error_message": "optional error description"
            }
        """
        print(f"🔄 开始获取 {exchange_name} 余额...")
        
        try:
            # 检查交易所是否支持
            if exchange_name not in EXCHANGES:
                return {
                    "exchange": exchange_name,
                    "balances": {},
                    "timestamp": int(time.time()),
                    "status": "error",
                    "error_message": f"Unsupported exchange: {exchange_name}"
                }
            
            # 加载API密钥
            load_api_keys_from_env()
            exchange_config = EXCHANGES[exchange_name]
            
            # 检查API密钥是否配置
            api_keys_missing = []
            for key, value in exchange_config["required_params"].items():
                # 跳过布尔类型的参数（如 use_vault）
                if isinstance(value, bool):
                    continue
                # 检查字符串类型的API密钥
                if not value or value == "":
                    api_keys_missing.append(key)
            
            if api_keys_missing:
                return {
                    "exchange": exchange_name,
                    "balances": {},
                    "timestamp": int(time.time()),
                    "status": "error",
                    "error_message": f"Missing API keys: {', '.join(api_keys_missing)}"
                }
            
            # 初始化交易所实例
            exchange = await self._initialize_exchange(exchange_name)
            if not exchange:
                return {
                    "exchange": exchange_name,
                    "balances": {},
                    "timestamp": int(time.time()),
                    "status": "error",
                    "error_message": "Failed to initialize exchange"
                }
            
            # 获取余额
            try:
                await exchange._update_balances()
                balances_dict = {}
                
                # 从交易所获取余额数据
                if hasattr(exchange, '_account_balances'):
                    for token, balance_info in exchange._account_balances.items():
                        if isinstance(balance_info, dict):
                            # 标准格式
                            available = float(balance_info.get('available', 0))
                            total = float(balance_info.get('total', balance_info.get('free', 0)))
                        else:
                            # 如果是Decimal对象
                            available = float(balance_info)
                            total = float(balance_info)
                        
                        # 只记录非零余额
                        if total > 0:
                            balances_dict[token] = {
                                "available": available,
                                "total": total
                            }
                
                print(f"✅ {exchange_name} 余额获取成功，共 {len(balances_dict)} 个币种")
                return {
                    "exchange": exchange_name,
                    "balances": balances_dict,
                    "timestamp": int(time.time()),
                    "status": "success"
                }
                
            except Exception as balance_error:
                print(f"❌ {exchange_name} 余额查询失败: {balance_error}")
                return {
                    "exchange": exchange_name,
                    "balances": {},
                    "timestamp": int(time.time()),
                    "status": "error",
                    "error_message": f"Balance query failed: {str(balance_error)}"
                }
            
        except Exception as e:
            print(f"❌ {exchange_name} 初始化失败: {e}")
            return {
                "exchange": exchange_name,
                "balances": {},
                "timestamp": int(time.time()),
                "status": "error",
                "error_message": f"Exchange initialization failed: {str(e)}"
            }
    
    async def _initialize_exchange(self, exchange_name: str):
        """初始化交易所实例"""
        try:
            if exchange_name in self.initialized_exchanges:
                return self.initialized_exchanges[exchange_name]
            
            exchange_config = EXCHANGES[exchange_name]
            exchange_class = exchange_config["exchange_class"]
            
            # 准备初始化参数
            init_params = {
                "client_config_map": self.client_config_adapter,
                "trading_pairs": [],  # 不需要交易对，只查询余额
                "trading_required": False  # 只读模式
            }
            
            # 添加API认证参数
            init_params.update(exchange_config["required_params"])
            
            # 创建交易所实例
            exchange = exchange_class(**init_params)
            
            # 缓存实例
            self.initialized_exchanges[exchange_name] = exchange
            
            return exchange
            
        except Exception as e:
            print(f"❌ Failed to initialize {exchange_name}: {e}")
            return None
    
    async def get_all_exchanges_balances(self) -> Dict[str, Any]:
        """获取所有支持交易所的余额信息"""
        print("🚀 开始获取所有交易所余额...")
        
        results = {}
        total_balances = {}
        
        # 并发获取所有交易所余额
        tasks = []
        for exchange_name in EXCHANGES.keys():
            task = asyncio.create_task(self.get_exchange_balances(exchange_name))
            tasks.append((exchange_name, task))
        
        # 等待所有任务完成
        for exchange_name, task in tasks:
            try:
                result = await task
                results[exchange_name] = result
                
                # 汇总余额
                if result["status"] == "success":
                    for token, balance in result["balances"].items():
                        if token not in total_balances:
                            total_balances[token] = {
                                "total_across_exchanges": 0,
                                "exchanges": {}
                            }
                        
                        total_balances[token]["total_across_exchanges"] += balance["total"]
                        total_balances[token]["exchanges"][exchange_name] = balance
                        
            except Exception as e:
                print(f"❌ Error getting balances for {exchange_name}: {e}")
                results[exchange_name] = {
                    "exchange": exchange_name,
                    "balances": {},
                    "timestamp": int(time.time()),
                    "status": "error",
                    "error_message": str(e)
                }
        
        return {
            "individual_exchanges": results,
            "aggregated_balances": total_balances,
            "timestamp": int(time.time()),
            "summary": {
                "total_exchanges": len(EXCHANGES),
                "successful_exchanges": sum(1 for r in results.values() if r["status"] == "success"),
                "failed_exchanges": sum(1 for r in results.values() if r["status"] == "error"),
                "total_unique_tokens": len(total_balances)
            }
        }
    
    async def cleanup(self):
        """清理资源"""
        for exchange_name, exchange in self.initialized_exchanges.items():
            try:
                if hasattr(exchange, 'stop'):
                    try:
                        # 尝试不带参数的stop()
                        await exchange.stop()
                    except TypeError:
                        try:
                            # 尝试带参数的stop()
                            exchange.stop()
                        except Exception:
                            # 如果都失败，尝试其他清理方法
                            if hasattr(exchange, '_stop'):
                                await exchange._stop()
                            elif hasattr(exchange, 'disconnect'):
                                await exchange.disconnect()
            except Exception as e:
                print(f"Warning: Failed to cleanup {exchange_name}: {e}")
        
        self.initialized_exchanges.clear()


class TestBalanceQuery(unittest.TestCase):
    """余额查询测试用例"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置 - 创建共享的事件循环"""
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.loop)
        cls.balance_service = BalanceQueryService()
    
    @classmethod
    def tearDownClass(cls):
        """类级别清理 - 清理共享资源"""
        try:
            cls.loop.run_until_complete(cls.balance_service.cleanup())
        except Exception as e:
            print(f"Warning during cleanup: {e}")
        finally:
            try:
                cls.loop.close()
            except Exception:
                pass
    
    def test_binance_balance(self):
        """测试Binance余额查询"""
        result = self.__class__.loop.run_until_complete(
            self.__class__.balance_service.get_exchange_balances("binance")
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["exchange"], "binance")
        self.assertIn("status", result)
        self.assertIn("balances", result)
        self.assertIn("timestamp", result)

        if result["status"] == "success":
            print(f"\n✅ BINANCE 交易所 - 成功查询到 {len(result['balances'])} 个币种:")
            for token, balance in result["balances"].items():
                print(f"   💰 {token}: {balance['total']:.6f} (可用: {balance['available']:.6f})")
        else:
            print(f"\n❌ BINANCE 交易所 - 查询失败:")
            print(f"   📝 错误: {result.get('error_message', 'Unknown error')}")
    
    def test_bybit_balance(self):
        """测试Bybit余额查询"""
        result = self.__class__.loop.run_until_complete(
            self.__class__.balance_service.get_exchange_balances("bybit")
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["exchange"], "bybit")
        self.assertIn("status", result)
        
        if result["status"] == "success":
            print(f"\n✅ BYBIT 交易所 - 成功查询到 {len(result['balances'])} 个币种:")
            for token, balance in result["balances"].items():
                print(f"   💰 {token}: {balance['total']:.6f} (可用: {balance['available']:.6f})")
        else:
            print(f"\n❌ BYBIT 交易所 - 查询失败:")
            print(f"   📝 错误: {result.get('error_message', 'Unknown error')}")
    
    def test_okx_balance(self):
        """测试OKX余额查询"""
        result = self.__class__.loop.run_until_complete(
            self.__class__.balance_service.get_exchange_balances("okx")
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["exchange"], "okx")
        self.assertIn("status", result)
        
        if result["status"] == "success":
            print(f"\n✅ OKX 交易所 - 成功查询到 {len(result['balances'])} 个币种:")
            for token, balance in result["balances"].items():
                print(f"   💰 {token}: {balance['total']:.6f} (可用: {balance['available']:.6f})")
        else:
            print(f"\n❌ OKX 交易所 - 查询失败:")
            print(f"   📝 错误: {result.get('error_message', 'Unknown error')}")
    
    def test_hyperliquid_balance(self):
        """测试Hyperliquid余额查询"""
        result = self.__class__.loop.run_until_complete(
            self.__class__.balance_service.get_exchange_balances("hyperliquid")
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["exchange"], "hyperliquid")
        self.assertIn("status", result)
        
        if result["status"] == "success":
            print(f"\n✅ HYPERLIQUID 交易所 - 成功查询到 {len(result['balances'])} 个币种:")
            for token, balance in result["balances"].items():
                print(f"   💰 {token}: {balance['total']:.6f} (可用: {balance['available']:.6f})")
        else:
            print(f"\n❌ HYPERLIQUID 交易所 - 查询失败:")
            print(f"   📝 错误: {result.get('error_message', 'Unknown error')}")
    
    def test_all_exchanges_balance(self):
        """测试所有交易所余额查询"""
        result = self.__class__.loop.run_until_complete(
            self.__class__.balance_service.get_all_exchanges_balances()
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("individual_exchanges", result)
        self.assertIn("aggregated_balances", result)
        self.assertIn("summary", result)
        
        summary = result["summary"]
        print(f"\n📊 余额查询汇总:")
        print(f"   总交易所数: {summary['total_exchanges']}")
        print(f"   成功查询: {summary['successful_exchanges']}")
        print(f"   查询失败: {summary['failed_exchanges']}")
        print(f"   总币种数: {summary['total_unique_tokens']}")
        
        # 显示聚合余额
        print(f"\n💰 跨交易所余额汇总:")
        for token, info in result["aggregated_balances"].items():
            total = info["total_across_exchanges"]
            exchanges_with_balance = len(info["exchanges"])
            print(f"   {token}: {total:.6f} (分布在 {exchanges_with_balance} 个交易所)")
    
    def test_invalid_exchange(self):
        """测试无效交易所处理"""
        result = self.__class__.loop.run_until_complete(
            self.__class__.balance_service.get_exchange_balances("invalid_exchange")
        )
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Unsupported exchange", result["error_message"])
    
    def test_missing_api_keys(self):
        """测试缺失API密钥的处理"""
        # 创建临时的服务实例进行测试
        temp_service = BalanceQueryService()
        
        # 模拟环境变量为空的情况
        with patch.dict(os.environ, {}, clear=True):
            result = self.__class__.loop.run_until_complete(
                temp_service.get_exchange_balances("binance")
            )
            
            # 应该返回错误状态
            self.assertEqual(result["status"], "error")
            self.assertIn("Missing API keys", result["error_message"])


async def main():
    """主函数 - 运行完整的余额查询测试"""
    print("🎯 FluxLayer 交易所余额查询测试")
    print("="*50)
    
    service = BalanceQueryService()
    
    try:
        # 获取所有交易所余额
        results = await service.get_all_exchanges_balances()
        
        print(f"\n📈 详细余额报告:")
        print(f"="*50)
        
        # 显示每个交易所的详细信息
        for exchange_name, exchange_result in results["individual_exchanges"].items():
            print(f"\n🏢 {exchange_name.upper()} 交易所:")
            if exchange_result["status"] == "success":
                balances = exchange_result["balances"]
                if balances:
                    print(f"   ✅ 状态: 成功查询到 {len(balances)} 个币种")
                    for token, balance in balances.items():
                        available = balance['available']
                        total = balance['total']
                        print(f"   💰 {token}: {total:.6f} (可用: {available:.6f})")
                else:
                    print(f"   💰 状态: 无余额或余额为零")
            else:
                print(f"   ❌ 状态: 查询失败")
                print(f"   📝 错误: {exchange_result.get('error_message', 'Unknown error')}")
        
        # 分析套利机会
        print(f"\n🔍 潜在套利机会分析:")
        print(f"="*30)
        aggregated = results["aggregated_balances"]
        for token, info in aggregated.items():
            if len(info["exchanges"]) >= 2:  # 存在于多个交易所
                print(f"🎯 {token} 存在于 {len(info['exchanges'])} 个交易所:")
                for ex_name, balance in info["exchanges"].items():
                    print(f"   - {ex_name}: {balance['total']:.6f}")
        
    except Exception as e:
        print(f"❌ 测试执行错误: {e}")
        import traceback
        print(traceback.format_exc())
    
    finally:
        await service.cleanup()
        print(f"\n✅ 余额查询测试完成!")


if __name__ == "__main__":
    # 检查是否在命令行直接运行
    if len(sys.argv) > 1 and sys.argv[1] == "--run-balance-query":
        # 运行完整的余额查询
        asyncio.run(main())
    else:
        # 运行单元测试
        print("🧪 运行余额查询单元测试...")
        print("使用 --run-balance-query 参数来运行实际的余额查询")
        unittest.main()