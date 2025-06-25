#!/usr/bin/env python3
"""
MCP 서버 - 네이버 뉴스 검색 & 주식 가격 조회
이 서버는 네이버 뉴스 검색 API와 yfinance를 사용하여 뉴스와 주식 정보를 제공합니다.
"""

import os
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

import requests
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastMCP 서버 초기화
mcp = FastMCP("news-stock-server")

# 상수 정의
NAVER_API_BASE = "https://openapi.naver.com/v1/search/news.json"

def is_valid_date(date_str: str) -> bool:
    """날짜 형식 검증 함수"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

class ToolException(Exception):
    """도구 실행 중 발생하는 예외"""
    pass

@mcp.tool()
async def naver_news_search(query: str, display: int = 10, start: int = 1, sort: str = "date") -> Dict[str, Any]:
    """
    네이버 검색 API를 사용하여 뉴스 검색 결과를 조회합니다.
    
    Args:
        query: 검색할 키워드
        display: 검색 결과 출력 건수 (1~100, 기본값: 10)
        start: 검색 시작 위치 (1~1000, 기본값: 1)
        sort: 정렬 옵션 (date: 날짜순, sim: 유사도순, 기본값: date)
    
    Returns:
        검색 결과와 상태 코드를 포함한 딕셔너리
    """
    
    # 환경변수에서 API 키 가져오기
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        return {
            "error": "네이버 API 키가 설정되지 않았습니다. NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET 환경변수를 설정해주세요.",
            "status_code": 400
        }
    
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    
    params = {
        "query": query,
        "display": min(max(display, 1), 100),  # 1-100 범위로 제한
        "start": min(max(start, 1), 1000),     # 1-1000 범위로 제한
        "sort": sort if sort in ["date", "sim"] else "date"
    }
    
    try:
        response = requests.get(NAVER_API_BASE, headers=headers, params=params, timeout=10)
        
        return {
            "data": response.json(),
            "status_code": response.status_code,
            "query_info": {
                "검색어": query,
                "출력건수": params["display"],
                "시작위치": params["start"],
                "정렬방식": params["sort"]
            }
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"네이버 API 요청 중 오류 발생: {e}")
        return {
            "error": f"API 요청 중 오류가 발생했습니다: {str(e)}",
            "status_code": 500
        }

@mcp.tool()
async def get_stock_price(symbol: str, date: Optional[str] = None, period: str = "5d") -> Dict[str, Any]:
    """
    yfinance를 사용하여 특정 날짜 또는 기간의 주식 가격 정보를 조회합니다.
    
    Args:
        symbol: 주식 심볼 (예: "AAPL", "TSLA", "005930.KS")
        date: 특정 날짜 (YYYY-MM-DD 형식, 선택사항)
        period: 조회 기간 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        주식 가격 정보를 포함한 딕셔너리
    """
    
    # 날짜 형식 검증
    if date and not is_valid_date(date):
        raise ToolException(f"잘못된 날짜 형식입니다: {date}. YYYY-MM-DD 형식을 사용해주세요.")
    
    try:
        stock = yf.Ticker(symbol)
        
        # 주식 정보 가져오기
        info = stock.info
        stock_name = info.get('longName', info.get('shortName', symbol))
        
        # 특정 날짜의 주식 가격 정보 조회
        if date:
            start_date = datetime.strptime(date, "%Y-%m-%d")
            end_date = start_date + timedelta(days=1)
            price_data = stock.history(start=start_date, end=end_date)
            
            # 가격 정보가 없으면 해당 날짜로부터 과거 5일간의 데이터 조회
            if price_data.empty:
                end_date = start_date
                start_date = start_date - timedelta(days=5)
                price_data = stock.history(start=start_date, end=end_date)
                
                if price_data.empty:
                    return {
                        "error": f"{symbol}에 대한 {date} 날짜의 데이터를 찾을 수 없습니다.",
                        "symbol": symbol,
                        "stock_name": stock_name,
                        "requested_date": date
                    }
        else:
            # 특정 날짜가 없으면 지정된 기간의 데이터 조회
            valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
            if period not in valid_periods:
                period = "5d"
            
            price_data = stock.history(period=period)
        
        if price_data.empty:
            return {
                "error": f"{symbol}에 대한 주식 데이터를 찾을 수 없습니다. 심볼을 확인해주세요.",
                "symbol": symbol,
                "stock_name": stock_name
            }
        
        # 데이터프레임을 딕셔너리로 변환
        df = price_data.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # 최신 데이터 (마지막 행)
        latest_data = df.iloc[-1].to_dict()
        
        # 전체 기간 데이터
        all_data = df.to_dict(orient='records')
        
        # 기본 주식 정보
        stock_info = {
            "symbol": symbol,
            "stock_name": stock_name,
            "currency": info.get('currency', 'USD'),
            "exchange": info.get('exchange', 'Unknown'),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": info.get('marketCap', 'Unknown'),
            "pe_ratio": info.get('trailingPE', 'Unknown'),
            "dividend_yield": info.get('dividendYield', 'Unknown')
        }
        
        result = {
            "stock_info": stock_info,
            "latest_price": latest_data,
            "period_data": all_data,
            "data_period": period if not date else f"around {date}",
            "total_records": len(all_data)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"주식 데이터 조회 중 오류 발생: {e}")
        raise ToolException(f"주식 데이터 조회 중 오류가 발생했습니다: {str(e)}")

@mcp.tool()
async def get_stock_comparison(symbols: list, period: str = "1mo") -> Dict[str, Any]:
    """
    여러 주식의 가격 정보를 비교합니다.
    
    Args:
        symbols: 주식 심볼 리스트 (예: ["AAPL", "MSFT", "GOOGL"])
        period: 조회 기간 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        여러 주식의 비교 정보를 포함한 딕셔너리
    """
    
    if not symbols or len(symbols) == 0:
        return {"error": "비교할 주식 심볼을 최소 1개 이상 입력해주세요."}
    
    if len(symbols) > 10:
        return {"error": "한 번에 최대 10개의 주식만 비교할 수 있습니다."}
    
    comparison_data = {}
    
    for symbol in symbols:
        try:
            result = await get_stock_price(symbol, period=period)
            if "error" not in result:
                comparison_data[symbol] = {
                    "name": result["stock_info"]["stock_name"],
                    "latest_price": result["latest_price"]["Close"],
                    "latest_date": result["latest_price"]["Date"],
                    "currency": result["stock_info"]["currency"],
                    "market_cap": result["stock_info"]["market_cap"],
                    "pe_ratio": result["stock_info"]["pe_ratio"]
                }
            else:
                comparison_data[symbol] = {"error": result["error"]}
        except Exception as e:
            comparison_data[symbol] = {"error": str(e)}
    
    return {
        "comparison_period": period,
        "stocks": comparison_data,
        "total_stocks": len(symbols),
        "successful_queries": len([s for s in comparison_data.values() if "error" not in s])
    }

@mcp.tool()
async def get_market_news_and_stock(query: str, stock_symbol: str) -> Dict[str, Any]:
    """
    특정 키워드로 뉴스를 검색하고 관련 주식 정보를 함께 조회합니다.
    
    Args:
        query: 검색할 뉴스 키워드
        stock_symbol: 조회할 주식 심볼
    
    Returns:
        뉴스 검색 결과와 주식 정보를 포함한 딕셔너리
    """
    
    # 뉴스 검색
    news_result = await naver_news_search(query, display=5)
    
    # 주식 정보 조회
    stock_result = await get_stock_price(stock_symbol)
    
    return {
        "query": query,
        "stock_symbol": stock_symbol,
        "news_data": news_result,
        "stock_data": stock_result,
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# 서버 실행
if __name__ == "__main__":
    # 환경변수 확인
    print("🚀 MCP 뉴스-주식 서버를 시작합니다...")
    
    # 네이버 API 키 확인
    if not os.getenv("NAVER_CLIENT_ID") or not os.getenv("NAVER_CLIENT_SECRET"):
        print("⚠️  경고: 네이버 API 키가 설정되지 않았습니다.")
        print("   NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET 환경변수를 설정해주세요.")
    else:
        print("✅ 네이버 API 키가 설정되었습니다.")
    
    print("📊 사용 가능한 도구:")
    print("   - naver_news_search: 네이버 뉴스 검색")
    print("   - get_stock_price: 주식 가격 조회")
    print("   - get_stock_comparison: 여러 주식 비교")
    print("   - get_market_news_and_stock: 뉴스 + 주식 통합 조회")
    
    # MCP 서버 실행
    mcp.run(transport='stdio')