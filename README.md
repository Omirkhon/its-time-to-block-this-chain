AI Crypto Assistant with Blockchain Integration
Overview
This project combines AI-powered document analysis with real-time cryptocurrency market data and blockchain-based vector operations. It features:

A conversational AI assistant that answers questions about both uploaded documents and cryptocurrency markets

Integration with multiple crypto data sources (CoinGecko, Binance, CryptoPanic)

Smart contract-powered vector mathematics on the Ethereum blockchain

Local LLM processing using Ollama

Vector document storage with ChromaDB

Features
1. Document Intelligence
Upload and process PDF, DOCX, and TXT files

AI-powered Q&A about document contents

Vector embeddings stored locally with ChromaDB

2. Crypto Market Assistant
Real-time price data from Binance API

Market cap and ranking from CoinGecko

Latest news from CryptoPanic

AI-generated summaries of market conditions

3. Blockchain Vector Operations
Smart contract for vector mathematics (addition, dot product, normalization)

Ethereum blockchain integration via Web3.py

Secure on-chain computations

Installation
Prerequisites
Python 3.8+

Ollama (for local LLM)

Ethereum node access (or Infura account)

API keys for crypto services

Setup
Clone the repository:

bash
git clone https://github.com/yourusername/ai-crypto-assistant.git
cd ai-crypto-assistant
Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Set up environment variables:

bash
cp .env.example .env
Edit .env with your API keys:

COINGECKO_API_KEY=your_key_here
CRYPTOPANIC_API_KEY=your_key_here
INFURA_PROJECT_ID=your_id_here
Download Ollama models:

bash
ollama pull llama2
Usage
Start the Streamlit app:

bash
streamlit run app.py
In the web interface:

Upload documents in the sidebar

Process documents when ready

Ask questions in the chat interface

Use the blockchain operations panel for vector math

Smart Contract Deployment
Compile the contract:

bash
solc --abi --bin VectorOperations.sol -o build/
Deploy to Ethereum:

python
from web3 import Web3

w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_INFURA_ID'))
with open('build/VectorOperations.abi') as f:
    abi = f.read()
with open('build/VectorOperations.bin') as f:
    bytecode = f.read()

contract = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = contract.constructor().transact()
tx_receipt = w3.eth.waitForTransactionReceipt(tx_hash)
print(f"Contract deployed at: {tx_receipt.contractAddress}")

API Requirements
You'll need accounts with:

CoinGecko API

CryptoPanic API

Binance API

Infura (for Ethereum access)

License
MIT License

Support
For issues or questions, please open an issue on GitHub.
