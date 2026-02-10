"""
도구 정의 모듈

OpenAI Function Calling에 사용할 도구들의 JSON 스키마를 정의합니다.
각 도구의 description은 LLM이 언제 해당 도구를 사용해야 할지 판단하는 데 매우 중요합니다.
명확하고 구체적인 description을 작성하여 LLM의 도구 선택 정확도를 높입니다.
"""

from typing import Dict, List, Optional

# ============================================================================
# 웹 검색 도구
# ============================================================================

SEARCH_WEB_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": """웹에서 최신 정보를 검색합니다.

이 도구를 사용해야 하는 경우:
- 실시간 정보 (주가, 날씨, 환율, 스포츠 결과 등)
- 최근 뉴스나 이벤트
- 특정 사실의 확인/검증
- "검색해줘", "찾아줘", "조사해줘" 등 명시적 요청
- 2024년 이후 정보
- 기업/인물/제품의 최신 현황
- 트렌드, 동향, 전망

이 도구를 사용하지 않아야 하는 경우:
- 일반 개념/정의 설명
- 프로그래밍 문법, 코드 작성
- 수학 계산, 논리적 추론
- 개인적 조언, 의견
- 창작물 (시, 소설, 이메일 등)
- 번역
- 확정된 역사적 사실""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 키워드나 질문. 구체적으로 작성. 예: 'Tesla stock 2024', 'AI agent trends'"
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "basic: 빠른 일반 검색, advanced: 심층 검색. 기본값 basic"
                }
            },
            "required": ["query"]
        }
    }
}

# ============================================================================
# 향후 추가될 도구들 (예시)
# ============================================================================

# FETCH_WEBPAGE_TOOL: Dict = {
#     "type": "function",
#     "function": {
#         "name": "fetch_webpage",
#         "description": "특정 웹페이지의 전체 내용을 가져옵니다.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "url": {
#                     "type": "string",
#                     "description": "가져올 웹페이지의 URL"
#                 }
#             },
#             "required": ["url"]
#         }
#     }
# }

# ============================================================================
# 도구 목록
# ============================================================================

AVAILABLE_TOOLS: List[Dict] = [
    SEARCH_WEB_TOOL,
    # 향후 추가될 도구들:
    # FETCH_WEBPAGE_TOOL,
]

# ============================================================================
# 도구 이름으로 찾기
# ============================================================================

TOOLS_BY_NAME: Dict[str, Dict] = {
    "search_web": SEARCH_WEB_TOOL,
    # 향후 추가될 도구들:
    # "fetch_webpage": FETCH_WEBPAGE_TOOL,
}

# ============================================================================
# 헬퍼 함수
# ============================================================================


def get_tool_by_name(name: str) -> Optional[Dict]:
    """
    도구 이름으로 도구 정의를 찾습니다.
    
    Args:
        name: 도구 이름 (예: "search_web")
    
    Returns:
        Optional[Dict]: 도구 정의 딕셔너리, 없으면 None
        
    Example:
        >>> tool = get_tool_by_name("search_web")
        >>> print(tool["function"]["name"])
        search_web
    """
    return TOOLS_BY_NAME.get(name)


def get_all_tool_names() -> List[str]:
    """
    사용 가능한 모든 도구 이름 리스트를 반환합니다.
    
    Returns:
        List[str]: 도구 이름 리스트
        
    Example:
        >>> names = get_all_tool_names()
        >>> print(names)
        ['search_web']
    """
    return list(TOOLS_BY_NAME.keys())
