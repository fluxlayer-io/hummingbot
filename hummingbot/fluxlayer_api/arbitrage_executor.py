import asyncio
import logging
import json
import os
import sys
from typing import Optional

import base58
import hashlib
import hmac

from hummingbot.fluxlayer_api.mpc_client import MPCClient
from solana.keypair import Keypair
from solana.publickey import PublicKey
from base64 import b64decode
import nacl.signing
import nacl.encoding
import bitcoin
from bitcoin import SelectParams
from bitcoin.wallet import CBitcoinSecret, CBitcoinAddress, P2PKHBitcoinAddress
from bitcoin.core import CTransaction, CMutableTransaction, CMutableTxIn, CMutableTxOut, COutPoint, lx, COIN, b2x, x
from bitcoin.core.script import CScript, OP_DUP, OP_HASH160, OP_EQUALVERIFY, OP_CHECKSIG, SignatureHash, SIGHASH_ALL
from bitcoin.core.scripteval import VerifyScript, SCRIPT_VERIFY_P2SH
import requests
from spl.token.client import Token
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import transfer_checked, TransferCheckedParams

# 基于 const.ts 的代币地址映射
TOKEN_MAPPINGS = {
    # Solana 主网代币
    "SOL": {
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "SOL": None  # 原生SOL
    },
    
    # Bitcoin 支持原生BTC
    "BTC": {
        "BTC": None  # 原生BTC
    },
    
    # 以太坊和其他EVM链的代币映射（来自const.ts）
    "ETH": {
        "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
        "DAI": "0x6b175474e89094c44da98b954eedeac495271d0f",
        "WBTC": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
        "FBTC": "0xc96de26018a54d51c097160568752c4e3bd6c364",
        "ETH": None  # 原生ETH
    },
    
    "BSC_BNB": {
        "USDC": "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d",
        "USDT": "0x55d398326f99059ff775485246999027b3197955",
        "BTCB": "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c",
        "FBTC": "0xc96de26018a54d51c097160568752c4e3bd6c364",
        "BNB": None  # 原生BNB
    },
    
    "ARBITRUM_ETH": {
        "USDC": "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
        "USDT": "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
        "FBTC": "0xc96de26018a54d51c097160568752c4e3bd6c364",
        "ETH": None  # 原生ETH
    }
}

# 简单 Logger 配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Fluxlayer Arbitrage Executor")

# ========== 简单私钥获取函数 ==========

def get_private_key_from_env(chain: str) -> Optional[str]:
    """从环境变量获取私钥"""
    env_vars = {
        "SOL": ["SOL_PRIVATE_KEY", "SOLANA_PRIVATE_KEY"],
        "BTC": ["BTC_PRIVATE_KEY", "BITCOIN_PRIVATE_KEY"],
        "ETH": ["ETH_PRIVATE_KEY", "ETHEREUM_PRIVATE_KEY"],
    }
    
    possible_vars = env_vars.get(chain, [f"{chain}_PRIVATE_KEY"])
    
    for var_name in possible_vars:
        private_key = os.getenv(var_name)
        if private_key and private_key.strip():
            logger.info(f"🔑 从环境变量 {var_name} 获取 {chain} 私钥")
            return private_key.strip()
    
    logger.error(f"❌ 未找到 {chain} 私钥，请设置环境变量：{' 或 '.join(possible_vars)}")
    return None

# Mock FluxLayer Exchange Metadata
class MockMetadata:
    def __init__(self):
        self.trading_pairs = {
            "USDC-BTC": MockPairMeta()
        }
        self.target_amount = 0.001
        self.source_amount = 0.1
        self.is_buy = True

class MockPairMeta:
    def __init__(self):
        self.source_chain = "SOL"
        self.target_chain = "BTC"
        self.source_token = "USDC"
        self.target_token = "BTC"

class MockFluxLayerExchange:
    def __init__(self):
        self.metadata = MockMetadata()

class ChainAdapter:
    def get_address_from_private_key(self, private_key: str) -> str:
        raise NotImplementedError
    
    async def transfer_token(self, private_key: str, to_address: str, amount: str, token: str) -> str:
        raise NotImplementedError
    
    def sign_message(self, private_key: str, message: bytes) -> str:
        raise NotImplementedError

class SolanaAdapter(ChainAdapter):
    
    def get_address_from_private_key(self, private_key: str) -> str:
        if not private_key:
            raise ValueError("Private key not provided")

        try:
            # 支持 Base58, Base64 和 JSON 数组格式
            if private_key.startswith("[") and private_key.endswith("]"):
                secret_key = bytes(json.loads(private_key))
            elif len(private_key) in [87, 88] and all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in private_key):
                secret_key = base58.b58decode(private_key)
            else:
                secret_key = b64decode(private_key)
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid private key format: {e}")

        if len(secret_key) not in [32, 64]:
            if len(secret_key) == 32:
                keypair = Keypair.from_seed(secret_key)
                return str(keypair.public_key)
            else:
                raise ValueError(f"Invalid private key length: {len(secret_key)} bytes, expected 32 or 64")

        keypair = Keypair.from_secret_key(secret_key)
        return str(keypair.public_key)
    
    def sign_message(self, private_key: str, message: bytes) -> str:
        try:
            # 解析私钥
            if private_key.startswith("[") and private_key.endswith("]"):
                secret = bytes(json.loads(private_key))
            elif len(private_key) in [87, 88] and all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in private_key):
                secret = base58.b58decode(private_key)
            else:
                secret = b64decode(private_key)
            
            if len(secret) not in [32, 64]:
                if len(secret) == 32:
                    keypair = Keypair.from_seed(secret)
                    secret = keypair.secret_key
                else:
                    raise ValueError(f"Invalid private key length: {len(secret)} bytes, expected 32 or 64")
            
            # 使用前32字节进行签名
            signing_secret = secret[:32] if len(secret) == 64 else secret
            signing_key = nacl.signing.SigningKey(signing_secret)
            signed = signing_key.sign(message)
            return signed.signature.hex()
            
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid private key format: {e}")
        except Exception as e:
            raise Exception(f"Signing failed: {e}")
    
    async def transfer_token(self, private_key: str, to_address: str, amount: str, token: str = "SOL") -> str:
        from solana.rpc.async_api import AsyncClient
        from solana.keypair import Keypair
        from solana.transaction import Transaction
        from solana.system_program import TransferParams, transfer
        from solana.publickey import PublicKey
        from solana.rpc.types import TxOpts
        
        # 使用官方主网端点
        rpc_endpoints = [
            "https://api.mainnet-beta.solana.com"
        ]
        
        client = None
        for endpoint in rpc_endpoints:
            try:
                client = AsyncClient(endpoint)
                logger.info(f"🔗 使用RPC端点: {endpoint}")
                break
            except Exception as e:
                logger.warning(f"⚠️ RPC端点 {endpoint} 连接失败: {e}")
                continue
        
        if not client:
            raise Exception("所有RPC端点都连接失败")
        
        # 使用全局代币映射
        TOKEN_MINTS = TOKEN_MAPPINGS.get("SOL", {})
        
        try:
            logger.info(f"🔄 转账: {amount} {token} 从私钥钱包到 {to_address}")
            
            # 解析私钥
            if len(private_key) in [87, 88]:
                secret = base58.b58decode(private_key)
            else:
                raise ValueError("私钥格式不正确，请使用Base58格式")
            
            sender = Keypair.from_secret_key(secret)
            recipient = PublicKey(to_address)
            logger.info(f"💳 发送方地址: {sender.public_key}")
            logger.info(f"🏦 接收方地址: {recipient}")
            
            # 获取latest blockhash
            import httpx
            
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getLatestBlockhash",
                    "params": [{"commitment": "processed"}]
                }
                
                rpc_url = client._provider.endpoint_uri
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.post(
                        rpc_url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    result = response.json()
                    logger.info(f"📡 Latest blockhash response: {result}")
                    
                    if 'result' in result and result['result'] is not None:
                        value = result['result']['value']
                        if 'blockhash' in value:
                            blockhash = value['blockhash']
                            logger.info(f"📡 获取到最新blockhash: {blockhash}")
                        else:
                            raise Exception(f"响应中没有blockhash: {value}")
                    else:
                        raise Exception(f"getLatestBlockhash请求失败: {result}")
                        
            except Exception as e:
                logger.error(f"❌ Failed to get latest blockhash: {e}")
                raise
            
            # 创建交易
            tx = Transaction(recent_blockhash=blockhash)
            
            if token == "SOL":
                # 原生SOL转账
                amount_lamports = int(float(amount) * 1_000_000_000)
                logger.info(f"💰 转账金额: {amount} SOL ({amount_lamports} lamports)")
                
                # 检查余额
                balance_resp = await client.get_balance(sender.public_key)
                if 'result' in balance_resp and balance_resp['result'] is not None:
                    balance = balance_resp['result']['value']
                    logger.info(f"💵 发送方余额: {balance / 1_000_000_000:.6f} SOL")
                else:
                    raise Exception(f"Failed to get balance: {balance_resp}")
                
                if balance < amount_lamports:
                    raise ValueError(f"余额不足: 需要 {amount_lamports}, 可用 {balance}")
                
                tx.add(transfer(TransferParams(
                    from_pubkey=sender.public_key,
                    to_pubkey=recipient,
                    lamports=amount_lamports
                )))
                
            else:
                # SPL代币转账
                if token not in TOKEN_MINTS:
                    raise ValueError(f"不支持的代币: {token}")
                
                mint_address = PublicKey(TOKEN_MINTS[token])
                decimals = 6 if token in ["USDC", "USDT"] else 6
                token_amount = int(float(amount) * (10 ** decimals))
                
                logger.info(f"💰 转账金额: {amount} {token} ({token_amount} 基础单位)")
                
                # 获取代币账户
                from solana.rpc.types import TokenAccountOpts
                
                sender_token_accounts = await client.get_token_accounts_by_owner(
                    sender.public_key, 
                    TokenAccountOpts(mint=mint_address)
                )
                
                if not sender_token_accounts.get('result', {}).get('value', []):
                    raise ValueError(f"发送方没有{token}代币账户")
                
                sender_token_account = PublicKey(sender_token_accounts['result']['value'][0]['pubkey'])
                
                # 获取或创建接收方关联代币账户 (Associated Token Account)
                from spl.token.instructions import get_associated_token_address, create_associated_token_account
                
                # 计算关联代币账户地址
                recipient_token_account = get_associated_token_address(recipient, mint_address)
                logger.info(f"🔍 计算的接收方ATA地址: {recipient_token_account}")
                
                # 检查接收方是否已有代币账户
                recipient_token_accounts = await client.get_token_accounts_by_owner(
                    recipient, 
                    TokenAccountOpts(mint=mint_address)
                )
                
                # 如果接收方没有代币账户，创建关联代币账户指令
                if not recipient_token_accounts.get('result', {}).get('value', []):
                    logger.info(f"⚠️ 接收方没有{token}代币账户，将自动创建关联代币账户")
                    
                    # 添加创建关联代币账户的指令
                    create_ata_ix = create_associated_token_account(
                        payer=sender.public_key,  # 发送方支付创建费用
                        owner=recipient,          # 接收方拥有账户
                        mint=mint_address
                    )
                    tx.add(create_ata_ix)
                    logger.info(f"✅ 已添加创建ATA指令")
                else:
                    # 如果已有账户，使用现有的
                    existing_account = recipient_token_accounts['result']['value'][0]['pubkey']
                    recipient_token_account = PublicKey(existing_account)
                    logger.info(f"✅ 使用现有代币账户: {recipient_token_account}")
                
                # 检查余额
                balance_info = await client.get_token_account_balance(sender_token_account)
                if 'result' in balance_info and balance_info['result'] is not None:
                    balance = int(balance_info['result']['value']['amount'])
                    balance_ui = float(balance_info['result']['value']['uiAmountString'])
                    logger.info(f"💵 发送方{token}余额: {balance_ui} {token}")
                else:
                    raise Exception(f"Failed to get token balance: {balance_info}")
                
                if balance < token_amount:
                    raise ValueError(f"{token}余额不足: 需要 {token_amount}, 可用 {balance}")
                
                tx.add(transfer_checked(TransferCheckedParams(
                    program_id=TOKEN_PROGRAM_ID,
                    source=sender_token_account,
                    mint=mint_address,
                    dest=recipient_token_account,
                    owner=sender.public_key,
                    amount=token_amount,
                    decimals=decimals,
                    signers=[sender.public_key]
                )))
            
            # 签名并发送交易
            tx.sign(sender)
            logger.info("✍️ 交易已签名")
            
            try:
                # 发送已签名的交易，使用原始字节形式
                logger.info("📤 准备发送交易...")
                logger.info(f"🔍 交易对象: {tx}")
                logger.info(f"🔍 交易指令数量: {len(tx.instructions)}")
                logger.info(f"🔍 交易blockhash: {tx.recent_blockhash}")
                
                # 序列化交易为字节
                tx_bytes = tx.serialize()
                logger.info(f"📦 交易字节长度: {len(tx_bytes)}")
                
                # 使用 send_raw_transaction 发送已序列化的交易
                import httpx
                
                # 手动构建 sendTransaction RPC 请求
                import base64
                
                rpc_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "sendTransaction",
                    "params": [
                        base64.b64encode(tx_bytes).decode('utf-8'),  # 交易的base64编码
                        {
                            "skipPreflight": False,
                            "preflightCommitment": "confirmed",
                            "encoding": "base64"
                        }
                    ]
                }
                
                logger.info("📤 发送交易到RPC节点...")
                
                rpc_url = client._provider.endpoint_uri
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.post(
                        rpc_url,
                        json=rpc_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    result = response.json()
                    logger.info(f"📡 交易发送响应: {result}")
                    
                    if 'result' in result and result['result'] is not None:
                        tx_signature = result['result']
                        logger.info(f"📤 交易已发送: {tx_signature}")
                    elif 'error' in result:
                        raise Exception(f"交易发送失败: {result['error']}")
                    else:
                        raise Exception(f"交易发送响应格式错误: {result}")
                        
            except Exception as send_error:
                logger.error(f"❌ 发送交易时出错: {send_error}")
                logger.error(f"❌ 错误类型: {type(send_error)}")
                import traceback
                traceback.print_exc()
                raise send_error
            
            # 简化确认过程 - 既然交易已经成功发送，我们等待几秒钟然后返回
            logger.info(f"⏳ 等待交易在链上确认: {tx_signature}")
            await asyncio.sleep(5)  # 等待5秒让交易上链
            
            logger.info(f"✅ 转账交易已发送: {tx_signature}")
            logger.info(f"🔍 可以在 Solana Explorer 查看交易: https://explorer.solana.com/tx/{tx_signature}")
            return tx_signature
            
        except Exception as e:
            logger.error(f"❌ 转账失败: {e}")
            return None
        finally:
            await client.close()

class BitcoinAdapter(ChainAdapter):
    """Bitcoin 区块链适配器 - 支持主网和测试网"""
    
    def __init__(self, network: str = "mainnet"):
        # 验证网络参数
        if network not in ["mainnet", "testnet"]:
            raise ValueError(f"不支持的网络: {network}，请使用 'mainnet' 或 'testnet'")
        
        self.network = network
        
        # 根据网络设置参数
        if network == "mainnet":
            bitcoin.SelectParams('mainnet')
            self.blockcypher_api = "https://api.blockcypher.com/v1/btc/main"
            self.explorer_url = "https://blockstream.info/tx"
        else:  # testnet
            bitcoin.SelectParams('testnet')
            self.blockcypher_api = "https://api.blockcypher.com/v1/btc/test3"
            self.explorer_url = "https://blockstream.info/testnet/tx"
        
        logger.info(f"🌐 Bitcoin适配器初始化: {network}")
    
    def get_address_from_private_key(self, private_key: str) -> str:
        """从WIF格式私钥获取Bitcoin地址"""
        if not private_key:
            raise ValueError("Private key not provided")
        
        try:
            # 验证并转换WIF私钥 - 根据网络验证前缀
            expected_prefixes = self._get_wif_prefixes()
            
            if (len(private_key) in [51, 52] and 
                any(private_key.startswith(prefix) for prefix in expected_prefixes)):
                # 使用 python-bitcoinlib 的 CBitcoinSecret
                private_key_obj = CBitcoinSecret(private_key)
                
                # 获取对应的地址 (P2PKH)
                address = P2PKHBitcoinAddress.from_pubkey(private_key_obj.pub)
                
                return str(address)
                
            else:
                expected_str = " 或 ".join(expected_prefixes)
                raise ValueError(f"无效的WIF私钥格式，{self.network}网络应以 {expected_str} 开头")
                
        except Exception as e:
            raise ValueError(f"Bitcoin私钥解析失败: {e}")
    
    def _get_wif_prefixes(self) -> list:
        """获取当前网络的WIF私钥前缀"""
        if self.network == "mainnet":
            return ['5', 'K', 'L']  # 主网前缀
        else:  # testnet
            return ['9', 'c']  # 测试网前缀
    
    def sign_message(self, private_key: str, message: bytes) -> str:
        """使用Bitcoin私钥签名消息"""
        try:
            private_key_obj = CBitcoinSecret(private_key)
            
            # 简化签名 - 直接对消息进行哈希并签名
            from bitcoin.core import Hash
            message_hash = Hash(message)
            
            # 使用私钥签名
            signature = private_key_obj.sign(message_hash)
            return signature.hex()
            
        except Exception as e:
            raise Exception(f"Bitcoin消息签名失败: {e}")
    
    async def _get_utxos(self, address: str):
        """获取地址的UTXO"""
        try:
            response = requests.get(f"{self.blockcypher_api}/addrs/{address}?unspentOnly=true")
            response.raise_for_status()
            data = response.json()
            
            if 'txrefs' not in data:
                return []
            
            utxos = []
            for utxo in data['txrefs']:
                utxos.append({
                    'txid': utxo['tx_hash'],
                    'vout': utxo['tx_output_n'],
                    'value': utxo['value'],  # satoshis
                    'scriptPubKey': utxo.get('script', '')
                })
            return utxos
            
        except Exception as e:
            logger.error(f"获取UTXO失败: {e}")
            return []
    
    async def _get_fee_rate(self):
        """获取当前推荐的交易费率 (satoshis per byte)"""
        try:
            response = requests.get(f"{self.blockcypher_api}")
            response.raise_for_status()
            data = response.json()
            
            # BlockCypher返回每KB的费用，转换为每字节
            fee_key = 'high_fee_per_kb' if self.network == "mainnet" else 'medium_fee_per_kb'
            default_fee = 50000 if self.network == "mainnet" else 20000
            
            return data.get(fee_key, default_fee) // 1000
            
        except Exception as e:
            logger.warning(f"获取费率失败，使用默认值: {e}")
            return 50 if self.network == "mainnet" else 20  # 测试网使用较低费率
    
    async def _broadcast_transaction(self, tx_hex: str):
        """广播交易到网络"""
        try:
            response = requests.post(
                f"{self.blockcypher_api}/txs/push",
                json={"tx": tx_hex}
            )
            response.raise_for_status()
            data = response.json()
            
            if 'tx' in data and 'hash' in data['tx']:
                return data['tx']['hash']
            else:
                raise Exception(f"广播响应格式错误: {data}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"广播交易失败: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"错误详情: {e.response.text}")
            raise Exception(f"交易广播失败: {e}")
    
    async def transfer_token(self, private_key: str, to_address: str, amount: str, token: str = "BTC") -> str:
        """执行Bitcoin转账"""
        try:
            # 支持的代币检查
            supported_tokens = ["BTC"]
            if self.network == "testnet":
                supported_tokens.append("XTN")  # testnet3上的原生代币
            
            if token not in supported_tokens:
                raise ValueError(f"Bitcoin适配器不支持代币: {token}，支持的代币: {supported_tokens}")
            
            # XTN在testnet3上等同于BTC的处理方式
            display_token = token
            logger.info(f"🔄 Bitcoin转账 ({self.network}): {amount} {display_token} 到 {to_address}")
            
            # 解析私钥
            private_key_obj = CBitcoinSecret(private_key)
            from_address = self.get_address_from_private_key(private_key)
            logger.info(f"💳 发送方地址: {from_address}")
            
            # 获取UTXO
            utxos = await self._get_utxos(from_address)
            if not utxos:
                raise ValueError(f"地址 {from_address} 没有可用的UTXO")
            
            # 计算金额（转换为satoshis）
            amount_satoshis = int(float(amount) * COIN)
            logger.info(f"💰 转账金额: {amount} {display_token} ({amount_satoshis} satoshis)")
            
            # 选择UTXO
            selected_utxos = []
            total_input = 0
            for utxo in utxos:
                selected_utxos.append(utxo)
                total_input += utxo['value']
                if total_input >= amount_satoshis:
                    break
            
            if total_input < amount_satoshis:
                raise ValueError(f"余额不足: 需要 {amount_satoshis}, 可用 {total_input}")
            
            # 获取费率并估算费用
            fee_rate = await self._get_fee_rate()
            estimated_size = len(selected_utxos) * 180 + 2 * 34 + 10  # 估算交易大小
            fee = estimated_size * fee_rate
            
            # 计算找零
            change = total_input - amount_satoshis - fee
            logger.info(f"💸 交易费用: {fee} satoshis ({fee_rate} sat/byte)")
            logger.info(f"🔄 找零: {change} satoshis")
            
            if change < 0:
                raise ValueError(f"余额不足支付费用: 总额 {total_input}, 转账 {amount_satoshis}, 费用 {fee}")
            
            # 构建交易输入
            txins = []
            for utxo in selected_utxos:
                txin = CMutableTxIn(COutPoint(lx(utxo['txid']), utxo['vout']))
                txins.append(txin)
            
            # 构建交易输出
            txouts = []
            
            # 主要输出 - 转账目标
            to_addr = CBitcoinAddress(to_address)
            txout_main = CMutableTxOut(amount_satoshis, to_addr.to_scriptPubKey())
            txouts.append(txout_main)
            
            # 找零输出（如果需要）
            dust_threshold = 546
            if change > dust_threshold:
                from_addr = P2PKHBitcoinAddress.from_pubkey(private_key_obj.pub)
                txout_change = CMutableTxOut(change, from_addr.to_scriptPubKey())
                txouts.append(txout_change)
            elif change > 0:
                logger.warning(f"⚠️ 找零 {change} satoshis 低于粉尘阈值 {dust_threshold}，将被包含在交易费中")
            
            # 创建未签名交易
            tx = CMutableTransaction(txins, txouts)
            logger.info("✍️ 创建未签名交易")
            
            # 签名每个输入
            for i, (txin, utxo) in enumerate(zip(txins, selected_utxos)):
                # 获取前一个输出的脚本（假设是P2PKH）
                from_addr = P2PKHBitcoinAddress.from_pubkey(private_key_obj.pub)
                scriptPubKey = from_addr.to_scriptPubKey()
                
                # 创建签名哈希
                sighash = SignatureHash(scriptPubKey, tx, i, SIGHASH_ALL)
                
                # 签名
                signature = private_key_obj.sign(sighash) + bytes([SIGHASH_ALL])
                
                # 创建解锁脚本
                scriptSig = CScript([signature, private_key_obj.pub])
                
                # 设置输入的解锁脚本
                txin.scriptSig = scriptSig
            
            logger.info("✍️ 交易已签名")
            
            # 序列化交易
            tx_serialized = tx.serialize()
            tx_hex = b2x(tx_serialized)
            
            # 广播交易
            tx_hash = await self._broadcast_transaction(tx_hex)
            logger.info(f"✅ {display_token}转账成功: {tx_hash}")
            logger.info(f"🔍 查看交易: {self.explorer_url}/{tx_hash}")
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"❌ Bitcoin转账失败: {e}")
            import traceback
            traceback.print_exc()
            return None

class ArbitrageExecutor:
    
    def __init__(self, network: str = "mainnet"):
        """
        初始化套利执行器
        
        Args:
            network: 网络类型，支持 "mainnet" 或 "testnet"
        """
        if network not in ["mainnet", "testnet"]:
            raise ValueError(f"不支持的网络: {network}，请使用 'mainnet' 或 'testnet'")
        
        self.network = network
        self.adapters = {
            "SOL": SolanaAdapter(),
            "BTC": BitcoinAdapter(network=network)
        }
    
    def get_adapter(self, chain: str) -> ChainAdapter:
        if chain not in self.adapters:
            raise ValueError(f"不支持的区块链: {chain}")
        return self.adapters[chain]
    
    async def execute_arbitrage(self, fluxlayer_exchange, trading_pair, private_key):
        try:
            pair_meta = fluxlayer_exchange.metadata.trading_pairs[trading_pair]
            i_amount = fluxlayer_exchange.metadata.source_amount
            o_amount = fluxlayer_exchange.metadata.target_amount
            is_buy = fluxlayer_exchange.metadata.is_buy
            # 根据 is_buy 决定交易方向
            if is_buy:
                # 买入操作：从 source 到 target
                src_chain = pair_meta.source_chain
                target_chain = pair_meta.target_chain
                i_token = pair_meta.source_token
                o_token = pair_meta.target_token
                direction = "BUY"
            else:
                # 卖出操作：从 target 到 source
                src_chain = pair_meta.target_chain
                target_chain = pair_meta.source_chain
                i_token = pair_meta.target_token
                o_token = pair_meta.source_token
                direction = "SELL"

            logger.info(f"🚀 开始套利执行 {trading_pair} ({direction})")
            logger.info(f"📊 源链: {src_chain} -> 目标链: {target_chain}")
            logger.info(f"💱 转换: {i_amount} {i_token} -> {o_amount} {o_token}")

            # 获取源链适配器
            src_adapter = self.get_adapter(src_chain)
            
            # 获取源地址
            src_addr = src_adapter.get_address_from_private_key(private_key)
            logger.info(f"👛 源地址 ({src_chain}): {src_addr}")

            # 初始化 MPC 客户端
            mpc_client = MPCClient()

            # 检查或创建 MPC 钱包
            logger.info("🔍 检查MPC钱包存在性...")
            if not mpc_client.check_mpc_exists(src_addr):
                logger.info(f"🆕 创建MPC钱包 for {src_addr}")
                wallet_id = mpc_client.create_mpc_wallet(src_addr)
                logger.info(f"✅ MPC钱包已创建，ID: {wallet_id}")
            else:
                wallet_id = mpc_client.get_mpc_wallet_id(src_addr)
                logger.info(f"✅ 找到现有MPC钱包，ID: {wallet_id}")

            # 获取 MPC 存款地址
            logger.info(f"🔍 查找MPC存款地址 for {src_chain}...")
            mpc_deposit_addr = mpc_client.find_mpc_addr(src_addr, src_chain)
            logger.info(f"🏦 MPC {src_chain} 存款地址: {mpc_deposit_addr}")
            
            # 验证 MPC 地址格式
            try:
                if src_chain == "SOL":
                    PublicKey(mpc_deposit_addr)
                    logger.info("✅ MPC地址格式有效")
            except Exception as e:
                logger.error(f"❌ 无效的MPC地址格式: {e}")
                return

            # 估算交易费用
            logger.info("💰 估算交易费用...")
            try:
                token_id = i_token
                if src_chain != i_token:
                    token_id = src_chain + "_" + i_token
                fee = mpc_client.estimate_tx_fee(wallet_id,token_id, i_amount, mpc_deposit_addr)
                total_deposit = float(i_amount) + float(fee)
                logger.info(f"💸 估算费用: {fee} {i_token}")
                logger.info(f"💰 总存款需求: {total_deposit} {i_token} ({i_amount} + {fee} 费用)")
            except Exception as e:
                logger.warning(f"⚠️ 费用估算失败: {e}, 使用默认费用")
                fee = 0.001 if i_token == "SOL" else 0
                total_deposit = float(i_amount) + fee

            # 执行代币转账到 MPC 地址
            logger.info(f"🔄 开始转账到MPC地址...")
            tx_hash = await src_adapter.transfer_token(
                private_key=private_key, 
                to_address=mpc_deposit_addr,
                amount=str(total_deposit),
                token=i_token
            )
            
            if not tx_hash:
                logger.error(f"❌ 转账到MPC地址失败")
                return
                
            logger.info(f"✅ 存款交易完成: {tx_hash}")

            # 签名目标订单
            logger.info("✍️ 签名maker订单...")
            try:
                message = f"{target_chain}{o_token}{o_amount}".encode("utf-8")
                sig = src_adapter.sign_message(private_key, message)
                logger.info(f"✅ 订单签名生成: {sig[:16]}...")
            except Exception as e:
                logger.error(f"❌ 订单签名失败: {e}")
                return

            # 创建maker订单
            logger.info("🔨 创建maker订单...")
            logger.info(f"📋 订单参数:")
            logger.info(f"   tx_hash: {tx_hash}")
            logger.info(f"   src_chain: {src_chain}")
            logger.info(f"   target_chain: {target_chain}")
            logger.info(f"   o_token: {o_token}")
            logger.info(f"   o_amount: {o_amount}")
            logger.info(f"   slippage: 0.01")
            logger.info(f"   sig: {sig[:32]}...")
            
            try:
                result = mpc_client.create_maker_order(
                    tx_hash=tx_hash,
                    src_chain=src_chain,
                    target_chain=target_chain,
                    o_token=o_token,
                    o_amount=str(o_amount),  # 确保是字符串
                    slippage="0.01",
                    sig=sig
                )
                logger.info(f"✅ Maker订单创建成功: {result}")
                
            except Exception as e:
                logger.error(f"❌ Maker订单创建失败: {e}")
                # 如果是requests异常，打印响应内容
                if hasattr(e, 'response'):
                    try:
                        error_detail = e.response.json()
                        logger.error(f"❌ 错误详情: {error_detail}")
                    except:
                        logger.error(f"❌ 响应内容: {e.response.text}")
                return

        except Exception as e:
            logger.error(f"💥 套利执行失败: {str(e)}")
            import traceback
            traceback.print_exc()

async def test_bitcoin_transfer(network: str = "testnet"):
    """测试Bitcoin转账功能"""
    try:
        logger.info(f"🧪 Bitcoin转账测试 ({network})")
        
        # 从环境变量获取私钥
        btc_private_key = get_private_key_from_env("BTC")
        
        if not btc_private_key:
            logger.error("❌ 无法获取BTC私钥，测试中止")
            logger.info(f"💡 {network}网络私钥格式:")
            if network == "testnet":
                logger.info("   - 以 '9' 或 'c' 开头 (测试网WIF格式)")
                logger.info("   - 获取测试币: https://coinfaucet.eu/en/btc-testnet/")
            else:
                logger.info("   - 以 'L', 'K' 或 '5' 开头 (主网WIF格式)")
            return
        
        # 获取接收地址
        recipient_address = os.getenv(f"BTC_RECIPIENT_{network.upper()}")
        if not recipient_address:
            logger.error(f"❌ 请设置环境变量 BTC_RECIPIENT_{network.upper()} 为接收地址")
            return
        
        logger.info(f"🔄 开始Bitcoin转账测试 ({network})...")
        
        btc_adapter = BitcoinAdapter(network=network)
        
        # 获取发送方地址
        from_address = btc_adapter.get_address_from_private_key(btc_private_key)
        logger.info(f"💳 Bitcoin发送方地址: {from_address}")
        
        # 执行转账 (转账0.001 BTC)
        tx_hash = await btc_adapter.transfer_token(
            private_key=btc_private_key,
            to_address=recipient_address,
            amount="0.001",
            token="BTC"
        )
        
        if tx_hash:
            logger.info(f"✅ Bitcoin转账成功: {tx_hash}")
            if network == "testnet":
                logger.info(f"🔍 查看交易: https://blockstream.info/testnet/tx/{tx_hash}")
            else:
                logger.info(f"🔍 查看交易: https://blockstream.info/tx/{tx_hash}")
        else:
            logger.error("❌ Bitcoin转账失败")
            
    except Exception as e:
        logger.error(f"💥 Bitcoin转账测试失败: {e}")
        import traceback
        traceback.print_exc()

async def execute_arbitrage_with_fluxlayer(trading_pair: str = "BTC-USDC"):
    fluxlayer_exchange = MockFluxLayerExchange()
    # 从环境变量获取Solana私钥
    private_key = get_private_key_from_env("SOL")
    
    if not private_key:
        logger.error("❌ 无法获取SOL私钥，套利执行中止")
        return
    
    # 创建套利执行器并执行
    executor = ArbitrageExecutor(network="mainnet")
    await executor.execute_arbitrage(fluxlayer_exchange, trading_pair, private_key)

async def test_xtn_transfer():
    """测试XTN代币转账功能（Bitcoin testnet3）"""
    try:
        logger.info("🧪 XTN转账测试 (Bitcoin testnet3)")
        
        # 从环境变量获取私钥
        btc_private_key = get_private_key_from_env("BTC")
        
        if not btc_private_key:
            logger.error("❌ 无法获取BTC私钥，测试中止")
            logger.info("💡 testnet网络私钥格式:")
            logger.info("   - 以 '9' 或 'c' 开头 (测试网WIF格式)")
            logger.info("   - 获取测试币: https://coinfaucet.eu/en/btc-testnet/")
            return
        
        # 获取接收地址
        recipient_address = os.getenv("BTC_RECIPIENT_TESTNET")
        if not recipient_address:
            logger.error("❌ 未找到接收地址环境变量 BTC_RECIPIENT_TESTNET")
            logger.info("💡 请设置环境变量: export BTC_RECIPIENT_TESTNET='你的测试网接收地址'")
            return
        
        # 初始化Bitcoin适配器 (testnet)
        bitcoin_adapter = BitcoinAdapter(network="testnet")
        
        # 获取发送地址
        from_address = bitcoin_adapter.get_address_from_private_key(btc_private_key)
        logger.info(f"👛 发送方地址: {from_address}")
        logger.info(f"🎯 接收方地址: {recipient_address}")
        
        # 转账金额
        amount = "0.001"  # 0.001 XTN
        
        logger.info(f"💰 转账金额: {amount} XTN")
        logger.info("🚀 开始XTN转账...")
        
        # 执行XTN转账
        tx_hash = await bitcoin_adapter.transfer_token(
            private_key=btc_private_key,
            to_address=recipient_address,
            amount=amount,
            token="XTN"  # 使用XTN代币
        )
        
        if tx_hash:
            logger.info(f"🎉 XTN转账成功完成!")
            logger.info(f"📝 交易哈希: {tx_hash}")
            logger.info(f"🔍 查看交易: https://blockstream.info/testnet/tx/{tx_hash}")
        else:
            logger.error("❌ XTN转账失败")
            
    except Exception as e:
        logger.error(f"💥 XTN转账测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    async def main():
        # 主程序
        await execute_arbitrage_with_fluxlayer()

    # 可以运行不同的测试
    # asyncio.run(test_bitcoin_transfer("testnet"))  # 测试BTC转账
    # asyncio.run(test_xtn_transfer())  # 测试XTN转账
    asyncio.run(main())  # 运行主程序