"""
대화 관리 모듈

[변경 이력]
- 2024-XX-XX: config.prompts에서 RESEARCH_ASSISTANT_SYSTEM_MESSAGE 임포트 추가
- 2024-XX-XX: __init__ 메서드의 system_message 파라미터 기본값을 
              RESEARCH_ASSISTANT_SYSTEM_MESSAGE로 설정하여 리서치 어시스턴트를 기본 역할로 사용
- 2024-XX-XX: 대화 상태 관리 기능 추가 (idle, responding, researching)
- 2024-XX-XX: 대화 저장/로드 기능 추가 (JSON 형식)
- 2024-XX-XX: 대화 요약 기능 추가
- 2024-XX-XX: chat 메서드에 재시도 로직 추가 (지수 백오프)
- 2024-XX-XX: 에러 처리 강화 (커스텀 예외 클래스, 구체적인 예외 타입, 로깅 개선)
- 2024-XX-XX: 하드코딩된 설정값을 config.settings 모듈로 분리
"""

import logging
from typing import List, Dict, Optional, Literal, Any
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv
import os
import json
import time
from datetime import datetime

# ============================================================================
# 커스텀 예외 클래스
# ============================================================================

class APIKeyNotFoundError(ValueError):
    """API 키가 설정되지 않은 경우 발생하는 예외
    
    OPENAI_API_KEY 환경변수가 설정되지 않았거나 .env 파일에
    API 키가 없는 경우 발생합니다.
    """
    pass


class ConversationSaveError(IOError):
    """대화 저장 중 오류가 발생한 경우 발생하는 예외
    
    파일 쓰기 권한 부족, 디스크 공간 부족, 파일 시스템 오류 등
    대화 저장 과정에서 발생하는 모든 I/O 관련 오류를 나타냅니다.
    """
    pass


class ConversationLoadError(IOError):
    """대화 로드 중 오류가 발생한 경우 발생하는 예외
    
    파일이 존재하지 않거나, JSON 형식이 올바르지 않거나,
    필수 필드가 없는 경우 등 대화 로드 과정에서 발생하는
    모든 오류를 나타냅니다.
    """
    pass


class ConversationSummaryError(Exception):
    """대화 요약 중 오류가 발생한 경우 발생하는 예외
    
    API 호출 실패, 응답 형식 오류 등 대화 요약 과정에서
    발생하는 모든 오류를 나타냅니다.
    """
    pass

# 리서치 어시스턴트 기본 시스템 메시지 임포트
# [변경] config.prompts 모듈에서 기본 시스템 메시지 임포트
try:
    from config.prompts import (
        RESEARCH_ASSISTANT_SYSTEM_MESSAGE,
        RESEARCH_ASSISTANT_SYSTEM_MESSAGE_V2,
        RESEARCH_MODE_PROMPT,
        RESPONDING_MODE_PROMPT,
    )
except ImportError:
    # config.prompts 모듈이 없는 경우를 위한 폴백
    RESEARCH_ASSISTANT_SYSTEM_MESSAGE = "당신은 도움이 되는 AI 어시스턴트입니다."
    RESEARCH_ASSISTANT_SYSTEM_MESSAGE_V2 = "당신은 웹 검색 기능을 갖춘 전문 리서치 어시스턴트입니다."
    RESEARCH_MODE_PROMPT = "[리서치 모드] 깊이 있는 분석과 구조화된 정보를 제공해주세요."
    RESPONDING_MODE_PROMPT = "[일반 응답 모드] 명확하고 간결하게 답변해주세요."

# 설정값 임포트
# [변경] config.settings 모듈에서 설정값 임포트
try:
    from config.settings import (
        DEFAULT_MODEL,
        DEFAULT_TEMPERATURE,
        RESEARCH_TEMPERATURE,
        SUMMARY_TEMPERATURE,
        MAX_TOKENS,
        RESEARCH_MAX_TOKENS,
        MAX_RETRIES,
        DATA_DIR,
        SAVE_FORMAT,
        BASE_BACKOFF_SECONDS,
        MIN_MESSAGES_FOR_SUMMARY,
    )
except ImportError:
    # config.settings 모듈이 없는 경우를 위한 폴백
    DEFAULT_MODEL: str = "gpt-4o-mini"
    DEFAULT_TEMPERATURE: float = 0.7
    RESEARCH_TEMPERATURE: float = 0.5
    SUMMARY_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1000
    RESEARCH_MAX_TOKENS: int = 1500
    MAX_RETRIES: int = 3
    DATA_DIR: str = "data"
    SAVE_FORMAT: str = "%Y%m%d_%H%M%S"
    BASE_BACKOFF_SECONDS: int = 2
    MIN_MESSAGES_FOR_SUMMARY: int = 3

# ============================================================================
# 내부 모듈 임포트 (Part 1 기능)
# ============================================================================

try:
    from src.search_agent import SearchAgent
    from src.memory_manager import MemoryManager
    from src.tools.tool_definitions import AVAILABLE_TOOLS
except ImportError:
    # Part 1 기능이 완성되지 않은 경우를 위한 폴백
    SearchAgent = None
    MemoryManager = None  # type: ignore[misc, assignment]
    AVAILABLE_TOOLS = []

# 환경변수 로드 (.env)
load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)

# 메시지 역할 타입 정의
MessageRole = Literal["system", "user", "assistant"]
MessageDict = Dict[Literal["role", "content"], str]

# 대화 상태 타입 정의
# [변경] 대화 상태 관리 기능 추가
ConversationState = Literal["idle", "responding", "researching"]

# 리서치 관련 키워드 목록 (상태 판단용)
RESEARCH_KEYWORDS: List[str] = ["조사", "분석", "리서치", "알아봐", "찾아봐"]

# [변경] 상수 정의를 config.settings로 이동
# 이제 모든 설정값은 config.settings 모듈에서 import하여 사용합니다.


class ConversationManager:
    """대화 세션을 관리하는 클래스
    
    OpenAI API를 사용하여 대화를 관리하고 히스토리를 저장합니다.
    대화 횟수를 추적하고 로깅 기능을 제공합니다.
    대화 상태를 관리하여 리서치 모드와 일반 응답 모드를 구분합니다.
    
    Attributes:
        client: OpenAI API 클라이언트 인스턴스
        messages: 대화 히스토리 리스트
        message_count: 대화 메시지 횟수 (사용자 입력 + AI 응답)
        state: 현재 대화 상태 ("idle", "responding", "researching")
    """
    
    def __init__(
        self,
        system_message: Optional[str] = None,
        enable_search: bool = True,
        memory_manager: Optional["MemoryManager"] = None,
        search_agent: Optional["SearchAgent"] = None,
    ) -> None:
        """ConversationManager 초기화
        
        OpenAI API 클라이언트를 초기화하고, 대화 히스토리와 상태를 설정합니다.
        system_message가 None인 경우 기본 리서치 어시스턴트 시스템 메시지를 사용합니다.
        enable_search가 True인 경우 웹 검색 기능을 활성화합니다.
        
        Args:
            system_message: 시스템 메시지 (선택적).
                기본값은 RESEARCH_ASSISTANT_SYSTEM_MESSAGE_V2 (웹 검색 기능 포함).
                None이 아닌 값이 제공되면 해당 메시지를 사용합니다.
                None이 제공되면 기본 리서치 어시스턴트 시스템 메시지를 사용합니다.
            enable_search: 웹 검색 기능 활성화 여부 (기본값: True).
                True인 경우 SearchAgent를 초기화하고 도구를 활성화합니다.
                False인 경우 검색 기능을 사용하지 않습니다.
            memory_manager: 메모리 관리자 (선택적). None이면 메모리 미사용.
            search_agent: 검색 에이전트 (선택적). None이면 enable_search 시 내부에서 생성.
        
        Raises:
            APIKeyNotFoundError: OPENAI_API_KEY 환경변수가 설정되지 않은 경우
        
        Note:
            기존 코드와의 호환성을 위해 None도 허용하되, None일 경우 기본값을 사용합니다.
            SearchAgent 초기화 실패 시에도 대화 기능은 정상적으로 동작합니다.
        """
        # system_message가 None인 경우 기본 리서치 어시스턴트 메시지 사용 (V2)
        if system_message is None:
            system_message = RESEARCH_ASSISTANT_SYSTEM_MESSAGE_V2
        
        # OpenAI API 클라이언트 초기화
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = (
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
                "다음 방법 중 하나로 API 키를 설정해주세요:\n"
                "1. .env 파일에 OPENAI_API_KEY=your_api_key 추가\n"
                "2. 환경변수로 직접 설정"
            )
            logger.error("API 키를 찾을 수 없습니다. .env 파일 또는 환경변수를 확인해주세요.")
            raise APIKeyNotFoundError(error_msg)
        
        self.client = OpenAI(api_key=api_key)
        
        # 대화 히스토리를 저장할 messages 리스트
        self.messages: List[MessageDict] = []
        
        # 대화 횟수 카운터 초기화
        self.message_count: int = 0
        
        # 대화 상태 초기화
        # [변경] 대화 상태 관리 기능 추가
        # 상태 종류: "idle" (대기), "responding" (일반 응답), "researching" (리서치 모드)
        self.state: ConversationState = "idle"
        
        # 시스템 메시지 추가
        # [변경] system_message는 이제 항상 값이 있으므로 (기본값 또는 사용자 제공값)
        #        조건문 없이 직접 추가
        self.messages.append({
            "role": "system",
            "content": system_message
        })
        logger.info(f"시스템 메시지 설정: {system_message[:50]}...")
        
        # 검색 기능 설정
        self.memory_manager = memory_manager
        self.enable_search = enable_search
        self.search_agent: Optional[SearchAgent] = None
        self.tools: Optional[List[dict]] = None
        
        if enable_search:
            try:
                if search_agent is not None:
                    self.search_agent = search_agent
                    self.tools = AVAILABLE_TOOLS if AVAILABLE_TOOLS else []
                    logger.info("검색 기능 활성화됨 (외부 SearchAgent 사용)")
                elif SearchAgent is not None:
                    self.search_agent = (
                        SearchAgent(memory_manager=memory_manager)
                        if memory_manager
                        else SearchAgent()
                    )
                    self.tools = AVAILABLE_TOOLS if AVAILABLE_TOOLS else []
                    logger.info("검색 기능 활성화됨")
                else:
                    logger.warning("SearchAgent가 사용 불가능합니다. 검색 기능을 비활성화합니다.")
                    self.enable_search = False
            except Exception as e:
                logger.warning(f"검색 기능 초기화 실패: {e}")
                self.enable_search = False
        if self.memory_manager:
            logger.info("ConversationManager initialized with memory")
        else:
            logger.info("ConversationManager initialized without memory")
        
        logger.info(
            f"ConversationManager 초기화 완료 "
            f"(search={'활성화' if self.enable_search else '비활성화'})"
        )
    
    def determine_state(self, user_input: str) -> ConversationState:
        """사용자 입력을 기반으로 대화 상태를 판단합니다.
        
        현재는 키워드 기반으로 상태를 판단하며, 리서치 관련 키워드가 포함된 경우
        "researching" 상태를 반환합니다. 그 외의 경우 "responding" 상태를 반환합니다.
        
        Args:
            user_input: 사용자의 입력 메시지
        
        Returns:
            ConversationState: 판단된 대화 상태
                - "researching": 리서치 관련 키워드가 포함된 경우
                - "responding": 일반 응답 모드
        
        Note:
            TODO: 4주차에 LLM 기반 판단으로 고도화 예정
        """
        # 입력을 소문자로 변환하여 대소문자 구분 없이 검색
        user_input_lower: str = user_input.lower()
        
        # 키워드 포함 여부 확인
        for keyword in RESEARCH_KEYWORDS:
            if keyword in user_input_lower:
                logger.debug(f"리서치 키워드 감지: '{keyword}' -> 상태: researching")
                return "researching"
        
        # 리서치 키워드가 없으면 일반 응답 모드
        return "responding"
    
    def _call_api_with_retry(
        self, 
        messages: List[MessageDict], 
        temperature: float,
        max_retries: int,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """OpenAI API를 호출하고 재시도 로직을 처리합니다.
        
        Args:
            messages: API에 전달할 메시지 리스트
            temperature: API 호출 시 사용할 temperature
            max_retries: 최대 재시도 횟수
            max_tokens: 최대 토큰 수 (기본값: MAX_TOKENS)
        
        Returns:
            str: AI 응답 메시지
        
        Raises:
            RateLimitError: API 요청 한도 초과 시
            APIConnectionError: 네트워크 연결 문제 시
            APIError: OpenAI API 호출 실패 시
            RuntimeError: 예상치 못한 오류로 재시도 후에도 실패한 경우
        """
        last_exception: Optional[Exception] = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # 시스템 메시지 확인 로깅
                    system_msg = next((msg for msg in messages if msg.get("role") == "system"), None)
                    if system_msg:
                        logger.debug(f"시스템 메시지 포함 확인: {system_msg['content'][:100]}...")
                    logger.debug(f"OpenAI API 호출 시작 (메시지 수: {len(messages)}, 모델: {DEFAULT_MODEL})")
                else:
                    wait_time: float = BASE_BACKOFF_SECONDS ** attempt
                    logger.info(
                        f"API 재시도 {attempt}/{max_retries} "
                        f"(메시지 수: {len(messages)}, 대기 시간: {wait_time}초)"
                    )
                    print(f"⚠ API 호출 실패. {attempt}번째 재시도 중... ({wait_time}초 대기)")
                
                response = self.client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # 응답 검증
                if not response.choices or len(response.choices) == 0:
                    error_msg = "API 응답에 선택지가 없습니다."
                    logger.error(error_msg)
                    raise APIError(error_msg)
                
                assistant_message: Optional[str] = response.choices[0].message.content
                
                if not assistant_message:
                    error_msg = "AI 응답이 비어있습니다."
                    logger.error(error_msg)
                    raise APIError(error_msg)
                
                if attempt > 0:
                    print(f"✓ 재시도 성공! 응답을 받았습니다.")
                
                return assistant_message
                
            except (RateLimitError, APIConnectionError, APIError) as e:
                last_exception = e
                
                if attempt < max_retries:
                    wait_time: float = BASE_BACKOFF_SECONDS ** attempt
                    logger.warning(
                        f"API 호출 실패 (시도 {attempt + 1}/{max_retries + 1}, "
                        f"예외 타입: {type(e).__name__}): {str(e)}. "
                        f"{wait_time}초 후 재시도..."
                    )
                    time.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                error_msg = f"예상치 못한 오류가 발생했습니다: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise Exception(error_msg) from e
        
        # 모든 재시도 실패 시 에러 처리
        if isinstance(last_exception, RateLimitError):
            error_msg = (
                f"API 요청 한도가 초과되었습니다.\n"
                f"{max_retries}번 재시도했지만 실패했습니다.\n"
                f"잠시 후 다시 시도해주세요."
            )
            logger.error(f"API 호출 최종 실패 (RateLimitError): {str(last_exception)}", exc_info=True)
            print(f"✗ {error_msg}")
            raise RateLimitError(error_msg) from last_exception
            
        elif isinstance(last_exception, APIConnectionError):
            error_msg = (
                f"네트워크 연결에 문제가 있습니다.\n"
                f"{max_retries}번 재시도했지만 실패했습니다.\n"
                f"인터넷 연결을 확인해주세요."
            )
            logger.error(f"API 호출 최종 실패 (APIConnectionError): {str(last_exception)}", exc_info=True)
            print(f"✗ {error_msg}")
            raise APIConnectionError(error_msg) from last_exception
            
        elif isinstance(last_exception, APIError):
            error_msg = (
                f"OpenAI API 호출 중 오류가 발생했습니다.\n"
                f"{max_retries}번 재시도했지만 실패했습니다.\n"
                f"오류 내용: {str(last_exception)}"
            )
            logger.error(f"API 호출 최종 실패 (APIError): {error_msg}", exc_info=True)
            print(f"✗ {error_msg}")
            raise APIError(error_msg) from last_exception
            
        else:
            error_msg = (
                f"API 호출이 {max_retries}번 재시도 후에도 실패했습니다.\n"
                f"원인을 파악할 수 없습니다."
            )
            logger.error(f"API 호출 최종 실패 (알 수 없는 오류): {error_msg}", exc_info=True)
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)
    
    def _call_api_with_tools(self) -> Any:
        """
        도구 정의를 포함하여 OpenAI API를 호출합니다.
        
        검색 기능이 활성화되어 있으면 tools 파라미터를 추가하여
        LLM이 필요시 도구를 호출할 수 있도록 합니다.
        
        Returns:
            OpenAI API 응답 객체
            
        Note:
            - tool_choice="auto": LLM이 도구 사용 여부를 자동 판단
            - tool_choice="required": 반드시 도구 사용
            - tool_choice="none": 도구 사용 안 함
        """
        # API 호출 파라미터 딕셔너리 구성
        call_params: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "messages": self.messages,
            "temperature": DEFAULT_TEMPERATURE,
        }
        
        # 검색 기능 활성화 시 도구 추가
        if self.enable_search and self.tools:
            call_params["tools"] = self.tools
            call_params["tool_choice"] = "auto"  # LLM이 자동 판단
            logger.debug(f"도구 포함 API 호출 (도구 개수: {len(self.tools)})")
        
        # API 호출 및 응답 반환
        response = self.client.chat.completions.create(**call_params)
        return response
    
    def _execute_tool(self, function_name: str, arguments: dict) -> str:
        """
        지정된 도구를 실행하고 결과를 반환합니다.
        
        Args:
            function_name: 실행할 도구 이름
            arguments: 도구에 전달할 인자 딕셔너리
        
        Returns:
            str: 도구 실행 결과 (LLM이 이해할 수 있는 형식)
                - 성공: format_for_llm() 결과 (마크다운 문자열)
                - 실패: JSON 문자열 {"error": "에러 메시지"}
        """
        try:
            # 1. search_web 도구 처리
            if function_name == "search_web":
                query = arguments.get("query", "")
                search_depth = arguments.get("search_depth", "basic")
                
                # 쿼리 검증
                if not query:
                    return json.dumps({"error": "검색어가 비어있습니다."})
                
                # SearchAgent가 없으면 에러 반환
                if not self.search_agent:
                    return json.dumps({"error": "검색 기능이 비활성화되어 있습니다."})
                
                # SearchAgent로 검색 실행
                search_result = self.search_agent.search(
                    query=query,
                    search_depth=search_depth
                )
                
                # LLM용 포맷으로 변환
                formatted = self.search_agent.format_for_llm(search_result)
                
                logger.info(
                    f"검색 완료: {search_result.result_count}개 결과, "
                    f"{search_result.search_time:.2f}초"
                )
                
                return formatted
            
            # 2. 알 수 없는 도구 처리
            else:
                logger.warning(f"알 수 없는 도구: {function_name}")
                return json.dumps({"error": f"알 수 없는 도구: {function_name}"})
        
        # 3. 예외 처리
        except Exception as e:
            logger.error(f"도구 실행 실패 ({function_name}): {str(e)}", exc_info=True)
            return json.dumps({"error": f"도구 실행 실패: {str(e)}"})
    
    def _handle_tool_calls(self, message) -> str:
        """
        LLM이 요청한 도구 호출을 처리합니다.
        
        1. Assistant의 도구 호출 요청을 메시지에 저장
        2. 각 도구를 실행하고 결과를 메시지에 추가
        3. 도구 결과를 포함하여 API를 다시 호출
        4. 최종 응답을 반환
        
        Args:
            message: OpenAI API 응답의 message 객체 (tool_calls 포함)
                - message.content: 텍스트 응답 (있을 수도 없을 수도 있음)
                - message.tool_calls: 도구 호출 리스트
        
        Returns:
            str: 도구 결과를 반영한 최종 AI 응답
        """
        # 1. Assistant 메시지 저장 (tool_calls 포함)
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": message.content,  # None일 수 있음
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        }
        self.messages.append(assistant_msg)
        
        # 2. 각 도구 호출 처리
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            logger.info(f"도구 실행: {function_name}({arguments})")
            
            # 도구 실행
            result = self._execute_tool(function_name, arguments)
            
            # 결과를 메시지에 추가 (tool role)
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        
        # 3. 도구 결과 포함하여 API 재호출
        final_response = self._call_api_with_tools()
        final_content = final_response.choices[0].message.content
        
        # 4. 최종 응답 저장 및 반환
        self.messages.append({
            "role": "assistant",
            "content": final_content
        })
        return final_content
    
    def chat(self, user_input: str, max_retries: int = MAX_RETRIES) -> str:
        """사용자 입력을 받아 AI 응답을 반환합니다.
        
        입력 검증, 상태 판단, 메시지 추가, API 호출(재시도 포함), 응답 처리의
        단계를 거쳐 AI 응답을 반환합니다. API 호출 실패 시 지수 백오프를 사용하여
        재시도합니다.
        
        Args:
            user_input: 사용자의 입력 메시지
            max_retries: API 호출 실패 시 최대 재시도 횟수. 기본값은 MAX_RETRIES(3)
            
        Returns:
            str: AI의 응답 메시지
            
        Raises:
            ValueError: 사용자 입력이 비어있는 경우
            RateLimitError: API 요청 한도 초과 시 (재시도 후에도 실패)
            APIConnectionError: 네트워크 연결 문제 시 (재시도 후에도 실패)
            APIError: OpenAI API 호출 실패 시 (재시도 후에도 실패)
            RuntimeError: 예상치 못한 오류로 재시도 후에도 실패한 경우
        
        Note:
            재시도 로직: 지수 백오프를 사용하여 2초, 4초, 8초 간격으로 재시도합니다.
        """
        # 1단계: 입력 검증
        if not user_input or not user_input.strip():
            error_msg = "사용자 입력이 비어있습니다. 메시지를 입력해주세요."
            logger.warning("빈 입력 검증 실패")
            raise ValueError(error_msg)
        
        # 1.5단계: 대화 상태 판단 및 업데이트
        # 사용자 입력을 분석하여 대화 상태를 판단하고 업데이트합니다.
        # TODO: 4주차에 LLM 기반 판단으로 고도화 예정
        previous_state: ConversationState = self.state
        self.state = self.determine_state(user_input)
        if previous_state != self.state:
            logger.info(f"대화 상태 변경: {previous_state} -> {self.state}")
        
        # 2단계: 상태에 따라 사용자 메시지 처리
        # 대화 히스토리에 사용자 메시지를 추가하여 컨텍스트 유지
        user_message: str = user_input.strip()
        
        # 상태에 따라 다른 프롬프트 추가
        if self.state == "researching":
            # 리서치 모드: 상세한 분석을 위한 프롬프트 추가
            enhanced_message = f"{RESEARCH_MODE_PROMPT}\n\n사용자 질문: {user_message}"
            logger.info(f"리서치 모드 활성화 - 상세 분석 프롬프트 추가")
        elif self.state == "responding":
            # 일반 응답 모드: 간결한 응답을 위한 프롬프트 추가
            enhanced_message = f"{RESPONDING_MODE_PROMPT}\n\n사용자 질문: {user_message}"
            logger.info(f"일반 응답 모드 - 간결한 응답 프롬프트 추가")
        else:
            # idle 상태 (이론적으로는 발생하지 않지만 안전을 위해)
            enhanced_message = user_message
        
        self.messages.append({
            "role": "user",
            "content": enhanced_message
        })
        self.message_count += 1
        logger.info(f"사용자 메시지 추가 (총 {self.message_count}개, 상태: {self.state}): {user_message[:50]}...")
        
        # 3단계: 도구 포함하여 OpenAI API 호출
        try:
            # 도구 포함하여 API 호출
            response = self._call_api_with_tools()
            message = response.choices[0].message
            
            # 도구 호출 확인 및 처리
            if message.tool_calls and len(message.tool_calls) > 0:
                logger.info(f"도구 호출 감지: {len(message.tool_calls)}개")
                result = self._handle_tool_calls(message)
            else:
                # 일반 응답
                result = message.content
                self.messages.append({
                    "role": "assistant",
                    "content": result
                })
            
            # 4단계: 메시지 카운트 증가 및 상태 업데이트
            self.message_count += 1
            logger.info(f"AI 응답 추가 (총 {self.message_count}개): {result[:50] if result else 'None'}...")
            
            # 5단계: 응답 완료 후 상태를 idle로 변경 (다음 입력 대기)
            self.state = "idle"
            logger.debug(f"응답 완료, 상태 변경: {self.state}")
            
            # ========== 메모리 자동 저장 ==========
            # [1] 검색 결과가 있으면 메모리에 저장 (도구 호출 시 search_results 전달되면 사용)
            _search_results = locals().get("search_results")
            if _search_results:
                try:
                    self.save_search_result_to_memory(
                        _search_results,
                        user_message
                    )
                except Exception as e:
                    logger.error(f"Failed to save search results: {e}")
            # [2] 대화 내용 메모리에 저장
            if self.memory_manager and result:
                try:
                    self.save_conversation_to_memory(
                        user_message,
                        result
                    )
                except Exception as e:
                    logger.error(f"Failed to save conversation: {e}")
            # ========== END 메모리 자동 저장 ==========
            
            return result
            
        except Exception as e:
            self.state = "idle"
            logger.error(f"응답 생성 실패: {str(e)}", exc_info=True)
            raise
    
    def get_messages(self) -> List[MessageDict]:
        """대화 히스토리를 반환합니다.
        
        원본 messages 리스트의 복사본을 반환하므로, 반환된 리스트를 수정해도
        원본에는 영향을 주지 않습니다.
        
        Returns:
            List[MessageDict]: 대화 히스토리 리스트의 복사본.
                각 항목은 {"role": str, "content": str} 형식입니다.
        """
        return self.messages.copy()
    
    def get_message_count(self) -> int:
        """대화 메시지 횟수를 반환합니다.
        
        사용자 입력과 AI 응답을 합한 총 메시지 횟수를 반환합니다.
        시스템 메시지는 카운트에 포함되지 않습니다.
        
        Returns:
            int: 현재까지의 대화 메시지 횟수 (사용자 입력 + AI 응답)
        """
        return self.message_count
    
    def get_state(self) -> ConversationState:
        """현재 대화 상태를 반환합니다.
        
        Returns:
            ConversationState: 현재 대화 상태
                - "idle": 대기 상태
                - "responding": 일반 응답 중
                - "researching": 리서치 모드
        """
        return self.state
    
    def _get_user_assistant_messages(self) -> List[MessageDict]:
        """시스템 메시지를 제외한 사용자/어시스턴트 메시지만 반환합니다.
        
        Returns:
            List[MessageDict]: 사용자와 어시스턴트 메시지 리스트
        """
        return [
            msg for msg in self.messages 
            if msg["role"] in ["user", "assistant"]
        ]
    
    # ============================================================================
    # 검색 관련 메서드 (2주차 추가)
    # ============================================================================
    
    def get_last_search_sources(self) -> List[str]:
        """
        마지막 검색의 출처 목록을 반환합니다.
        
        Returns:
            List[str]: 출처 URL 목록, 검색 기록 없으면 빈 리스트
        """
        if self.search_agent:
            return self.search_agent.get_sources()
        return []
    
    def get_search_count(self) -> int:
        """
        총 검색 횟수를 반환합니다.
        
        Returns:
            int: 검색 횟수
        """
        if self.search_agent:
            return self.search_agent.get_search_count()
        return 0
    
    def is_search_enabled(self) -> bool:
        """
        검색 기능 활성화 여부를 반환합니다.
        
        Returns:
            bool: 검색 기능 활성화 여부
        """
        return self.enable_search
    
    def save_search_result_to_memory(
        self, 
        search_results: dict, 
        user_query: str
    ) -> int:
        """
        검색 결과를 메모리에 저장
        
        Args:
            search_results: search_with_memory()의 반환값
            user_query: 사용자 검색 쿼리
        
        Returns:
            저장된 결과 수
        """
        if not self.memory_manager:
            return 0
        
        saved_count = 0
        
        # 병합된 결과에서 상위 5개만 저장
        merged_results = search_results.get("merged_results", [])
        
        for result in merged_results[:5]:
            try:
                # 이미 메모리에 있는 결과는 스킵
                if result.get("source") == "memory":
                    continue
                
                # 메타데이터 구성
                metadata = {
                    "source": "search_result",
                    "user_query": user_query,
                    "original_source": result.get("source", "unknown"),
                    "saved_from": "conversation"
                }
                
                # URL이 있으면 추가
                if "url" in result:
                    metadata["url"] = result["url"]
                
                # 메모리에 저장
                self.memory_manager.add_to_memory(
                    text=result["content"],
                    metadata=metadata,
                    check_duplicate=True
                )
                
                saved_count += 1
                logger.debug(f"Saved search result to memory")
                
            except Exception as e:
                logger.error(f"Failed to save search result: {e}")
                continue
        
        if saved_count > 0:
            logger.info(f"Saved {saved_count} search results to memory")
        
        return saved_count
    
    def save_conversation_to_memory(
        self, 
        user_message: str, 
        assistant_message: str
    ) -> bool:
        """
        대화 내용을 요약하여 메모리에 저장
        
        Args:
            user_message: 사용자 메시지
            assistant_message: AI 응답
        
        Returns:
            저장 성공 여부
        """
        if not self.memory_manager:
            return False
        
        try:
            # 대화 요약 생성 (간단한 버전)
            conversation_summary = f"""
사용자 질문: {user_message}

AI 응답: {assistant_message[:300]}{"..." if len(assistant_message) > 300 else ""}
            """.strip()
            
            # 메타데이터 구성
            metadata = {
                "source": "conversation",
                "user_query": user_message,
                "response_length": len(assistant_message),
                "saved_from": "chat_history"
            }
            
            # 메모리에 저장
            self.memory_manager.add_to_memory(
                text=conversation_summary,
                metadata=metadata,
                check_duplicate=True
            )
            
            logger.info("Saved conversation to memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False
    
    def _summarize_conversation_with_gpt(
        self, 
        user_message: str, 
        assistant_message: str
    ) -> str:
        """GPT로 대화를 간결하게 요약 (선택적)"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "다음 대화를 한 문장으로 요약해주세요."
                    },
                    {
                        "role": "user",
                        "content": f"사용자: {user_message}\nAI: {assistant_message}"
                    }
                ],
                max_tokens=100
            )
            
            summary = response.choices[0].message.content
            return summary
            
        except Exception as e:
            logger.error(f"GPT summarization failed: {e}")
            # 실패 시 기본 요약 사용
            return f"{user_message} - {assistant_message[:100]}..."
    
    def clear_history(self) -> None:
        """대화 히스토리를 초기화합니다.
        
        시스템 메시지는 유지하고, 사용자와 AI의 대화만 삭제합니다.
        대화 상태도 "idle"로 초기화하고, message_count도 0으로 리셋합니다.
        
        Note:
            시스템 메시지는 유지되므로, 초기화 후에도 동일한 시스템 프롬프트가
            적용된 상태로 대화를 계속할 수 있습니다.
        """
        system_messages: List[MessageDict] = [
            msg for msg in self.messages if msg["role"] == "system"
        ]
        self.messages = system_messages
        self.message_count = 0
        # [변경] 대화 상태 관리 기능 추가 - 히스토리 초기화 시 상태도 초기화
        self.state = "idle"
        
        # 검색 히스토리도 초기화
        if self.search_agent:
            self.search_agent.clear_history()
        
        logger.info("대화 히스토리 초기화됨")
    
    def summarize_conversation(self) -> str:
        """현재 대화 히스토리를 요약합니다.
        
        시스템 메시지를 제외한 실제 대화 메시지가 MIN_MESSAGES_FOR_SUMMARY개 이하면
        요약하지 않고 안내 메시지를 반환합니다. 요약은 임시 메시지로 API를 호출하므로
        실제 messages에는 추가되지 않습니다.
        
        Returns:
            str: 대화 요약 텍스트 또는 "대화가 충분히 길지 않습니다" 메시지
        
        Raises:
            ConversationSummaryError: 요약 처리 중 오류 발생 시
            RateLimitError: API 요청 한도 초과 시
            APIConnectionError: 네트워크 연결 문제 시
            APIError: OpenAI API 호출 실패 시
        
        Note:
            대화가 MIN_MESSAGES_FOR_SUMMARY개 이하면 요약하지 않고 안내 메시지를 반환합니다.
        """
        # 1단계: 대화 길이 확인
        # 시스템 메시지를 제외한 실제 대화 메시지 수 확인
        conversation_messages: List[MessageDict] = self._get_user_assistant_messages()
        
        if len(conversation_messages) <= MIN_MESSAGES_FOR_SUMMARY:
            logger.info(f"대화가 충분히 길지 않아 요약을 건너뜁니다. (현재: {len(conversation_messages)}개, 최소: {MIN_MESSAGES_FOR_SUMMARY + 1}개 필요)")
            return "대화가 충분히 길지 않습니다"
        
        try:
            # 2단계: 임시 메시지 리스트 생성 (요약 요청 추가)
            # messages를 복사하여 요약 요청을 추가 (원본 messages는 변경하지 않음)
            temp_messages: List[MessageDict] = self.messages.copy()
            temp_messages.append({
                "role": "user",
                "content": "지금까지의 대화를 3문장으로 요약해주세요."
            })
            
            # 3단계: OpenAI API 호출하여 요약 생성
            # SUMMARY_TEMPERATURE 사용 (일관성을 위해 낮게 설정)
            logger.debug(f"대화 요약 API 호출 시작 (메시지 수: {len(temp_messages)})")
            summary: str = self._call_api_with_retry(
                messages=temp_messages,
                temperature=SUMMARY_TEMPERATURE,
                max_retries=MAX_RETRIES
            )
            
            logger.info(
                f"대화 요약 완료 (원본 메시지 수: {len(self.messages)}, "
                f"대화 메시지 수: {len(conversation_messages)}, 요약 길이: {len(summary)}자)"
            )
            return summary
            
        except (RateLimitError, APIConnectionError, APIError):
            # API 관련 예외는 그대로 전파
            raise
        except Exception as e:
            error_msg = f"대화 요약 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConversationSummaryError(error_msg) from e
    
    def save_conversation(self, filename: Optional[str] = None) -> None:
        """대화 히스토리를 JSON 파일로 저장합니다.
        
        DATA_DIR 폴더에 JSON 형식으로 저장하며, filename이 None이면
        타임스탬프 기반 파일명을 자동 생성합니다. 저장되는 데이터에는
        timestamp, messages, message_count, state가 포함됩니다.
        
        Args:
            filename: 저장할 파일명 (선택적).
                None이면 타임스탬프 기반 파일명 자동 생성.
                예: "conversation_20240115_143022.json"
        
        Raises:
            ConversationSaveError: 파일 저장 중 오류 발생 시
            PermissionError: 파일 쓰기 권한이 없는 경우
            OSError: 디렉토리 생성 실패 등 시스템 오류 발생 시
        
        Note:
            DATA_DIR 폴더가 없으면 자동으로 생성합니다.
        """
        try:
            # data/ 폴더 생성 (존재하지 않는 경우)
            if not os.path.exists(DATA_DIR):
                try:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    logger.info(f"'{DATA_DIR}' 폴더를 생성했습니다.")
                except OSError as e:
                    error_msg = f"'{DATA_DIR}' 폴더를 생성할 수 없습니다: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise ConversationSaveError(error_msg) from e
            
            # 파일명 생성
            if filename is None:
                # 타임스탬프 기반 파일명 생성
                timestamp: str = datetime.now().strftime(SAVE_FORMAT)
                filename = f"conversation_{timestamp}.json"
            
            # 파일 경로 생성
            filepath: str = os.path.join(DATA_DIR, filename)
            
            # 저장할 데이터 구성
            save_data: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "messages": self.messages,
                "message_count": self.message_count,
                "state": self.state,
                # 검색 관련 정보 추가
                "search_enabled": self.enable_search,
                "search_count": self.get_search_count()
            }
            
            # JSON 파일로 저장
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
            except PermissionError as e:
                error_msg = f"파일 쓰기 권한이 없습니다: {filepath}"
                logger.error(f"{error_msg} 원본 오류: {str(e)}")
                raise ConversationSaveError(error_msg) from e
            except IOError as e:
                error_msg = f"파일 저장 중 I/O 오류가 발생했습니다: {filepath}"
                logger.error(f"{error_msg} 원본 오류: {str(e)}", exc_info=True)
                raise ConversationSaveError(error_msg) from e
            
            logger.info(f"대화가 저장되었습니다: {filepath} (메시지 수: {len(self.messages)}, 대화 횟수: {self.message_count})")
            print(f"✓ 대화가 저장되었습니다: {filepath}")
            
        except (ConversationSaveError, PermissionError, OSError):
            # 이미 처리된 예외는 그대로 전파
            raise
        except Exception as e:
            error_msg = f"대화 저장 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConversationSaveError(error_msg) from e
    
    def load_conversation(self, filename: str) -> None:
        """저장된 대화 히스토리를 JSON 파일에서 로드합니다.
        
        DATA_DIR 폴더에서 JSON 파일을 로드하여 messages, message_count,
        state를 복원합니다. message_count나 state가 없는 경우 자동으로
        계산하거나 기본값으로 초기화합니다.
        
        Args:
            filename: 로드할 파일명 (DATA_DIR 폴더 내 파일명)
        
        Raises:
            ConversationLoadError: 파일 로드 중 오류 발생 시
            FileNotFoundError: 파일이 존재하지 않는 경우
            PermissionError: 파일 읽기 권한이 없는 경우
            json.JSONDecodeError: JSON 파싱 오류 발생 시
            ValueError: 필수 필드가 없는 경우
        
        Note:
            - message_count가 없으면 자동 계산합니다.
            - state가 없으면 "idle"로 초기화합니다.
        """
        try:
            # 파일 경로 생성
            filepath: str = os.path.join(DATA_DIR, filename)
            
            # 파일 존재 확인
            if not os.path.exists(filepath):
                error_msg = (
                    f"파일을 찾을 수 없습니다: {filepath}\n"
                    f"'{DATA_DIR}' 폴더 내의 파일명을 확인해주세요."
                )
                logger.error(f"파일을 찾을 수 없음: {filepath}")
                raise FileNotFoundError(error_msg)
            
            # JSON 파일 읽기
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    load_data: Dict[str, Any] = json.load(f)
            except PermissionError as e:
                error_msg = f"파일 읽기 권한이 없습니다: {filepath}"
                logger.error(f"{error_msg} 원본 오류: {str(e)}")
                raise ConversationLoadError(error_msg) from e
            except IOError as e:
                error_msg = f"파일 읽기 중 I/O 오류가 발생했습니다: {filepath}"
                logger.error(f"{error_msg} 원본 오류: {str(e)}", exc_info=True)
                raise ConversationLoadError(error_msg) from e
            except json.JSONDecodeError as e:
                error_msg = (
                    f"JSON 파일 형식이 올바르지 않습니다: {filepath}\n"
                    f"파일이 손상되었거나 올바른 JSON 형식이 아닙니다."
                )
                logger.error(f"JSON 파싱 오류: {filepath}, 오류: {str(e)}", exc_info=True)
                raise ConversationLoadError(error_msg) from e
            
            # 필수 필드 검증
            if "messages" not in load_data:
                error_msg = (
                    f"JSON 파일에 필수 필드 'messages'가 없습니다: {filepath}\n"
                    f"파일이 올바른 형식인지 확인해주세요."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # messages 복원
            try:
                self.messages = load_data["messages"]
                if not isinstance(self.messages, list):
                    raise ValueError("'messages' 필드는 리스트여야 합니다.")
            except (TypeError, ValueError) as e:
                error_msg = f"메시지 데이터 형식이 올바르지 않습니다: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ConversationLoadError(error_msg) from e
            
            # message_count 복원 (있는 경우)
            if "message_count" in load_data:
                try:
                    self.message_count = int(load_data["message_count"])
                except (TypeError, ValueError) as e:
                    logger.warning(f"message_count 형식이 올바르지 않아 자동 계산합니다. 원본 오류: {str(e)}")
                    user_assistant_messages: List[MessageDict] = self._get_user_assistant_messages()
                    self.message_count = len(user_assistant_messages)
            else:
                # message_count가 없으면 messages 길이에서 계산
                # 시스템 메시지 제외하고 계산
                user_assistant_messages: List[MessageDict] = self._get_user_assistant_messages()
                self.message_count = len(user_assistant_messages)
                logger.info("message_count가 없어 자동 계산했습니다.")
            
            # state 복원 (있는 경우)
            if "state" in load_data:
                state_value = load_data["state"]
                if state_value in ["idle", "responding", "researching"]:
                    self.state = state_value
                else:
                    logger.warning(f"state 값이 올바르지 않아 'idle'로 초기화합니다. (받은 값: {state_value})")
                    self.state = "idle"
            else:
                self.state = "idle"
                logger.info("state가 없어 'idle'로 초기화했습니다.")
            
            logger.info(
                f"대화가 로드되었습니다: {filepath} "
                f"(메시지 수: {len(self.messages)}, 대화 횟수: {self.message_count}, 상태: {self.state})"
            )
            print(f"✓ 대화가 로드되었습니다: {filepath}")
            print(f"  - 메시지 수: {len(self.messages)}개")
            print(f"  - 대화 횟수: {self.message_count}회")
            print(f"  - 현재 상태: {self.state}")
            
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, ValueError):
            # 구체적인 예외는 그대로 전파
            raise
        except ConversationLoadError:
            # 커스텀 예외도 그대로 전파
            raise
        except Exception as e:
            error_msg = f"대화 로드 중 예상치 못한 오류가 발생했습니다: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"✗ 오류: {error_msg}")
            raise ConversationLoadError(error_msg) from e


# ---------------------------------------------------------------------------
# 테스트 코드 (주석) — ConversationManager + MemoryManager 연동 예시
# ---------------------------------------------------------------------------
# from src.conversation_manager import ConversationManager
# from src.search_agent import SearchAgent
# from src.memory_manager import MemoryManager
#
# # MemoryManager 초기화
# mm = MemoryManager("conv_memory", "data/chroma_db")
#
# # SearchAgent 초기화 (메모리 포함)
# agent = SearchAgent(memory_manager=mm)
#
# # ConversationManager 초기화 (메모리 포함)
# conv_mgr = ConversationManager(
#     search_agent=agent,
#     memory_manager=mm
# )
#
# print(f"Memory enabled: {conv_mgr.memory_manager is not None}")


# ---------------------------------------------------------------------------
# 테스트 코드 (주석) — 검색 결과 메모리 저장 예시
# ---------------------------------------------------------------------------
# # 검색 수행
# agent = SearchAgent(memory_manager=mm)
# search_results = agent.search_with_memory("테슬라")
# 
# # ConversationManager로 저장
# conv_mgr = ConversationManager(agent, mm)
# saved = conv_mgr.save_search_result_to_memory(
#     search_results, 
#     "테슬라"
# )
# 
# print(f"저장된 결과: {saved}개")


# ---------------------------------------------------------------------------
# 테스트 코드 (주석) — 대화 내용 메모리 저장 예시
# ---------------------------------------------------------------------------
# conv_mgr = ConversationManager(agent, mm)
#
# # 대화 저장
# success = conv_mgr.save_conversation_to_memory(
#     user_message="테슬라에 대해 알려줘",
#     assistant_message="테슬라는 2003년 설립된..."
# )
#
# print(f"저장 성공: {success}")
#
# # 메모리에서 검색
# results = mm.search_memory("테슬라", top_k=3)
# for r in results:
#     if r['metadata'].get('source') == 'conversation':
#         print(f"대화 내용: {r['text'][:50]}...")
