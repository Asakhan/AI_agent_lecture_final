"""
BaseAgent 추상 클래스

모든 에이전트가 상속하는 기본 클래스. LLM 호출 공통 로직과 execute 인터페이스를 정의합니다.
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from openai import OpenAI

from config.settings import DEFAULT_MODEL, MAX_TOKENS


class BaseAgent(ABC):
    """
    에이전트 추상 기본 클래스.

    OpenAI 클라이언트를 사용하여 LLM을 호출하고, 하위 클래스는 execute()를 구현합니다.
    """

    def __init__(
        self,
        client: OpenAI,
        name: str,
        role: str,
        system_prompt: str,
    ) -> None:
        """
        Args:
            client: OpenAI API 클라이언트 인스턴스
            name: 에이전트 식별 이름
            role: 에이전트 역할 설명
            system_prompt: LLM 시스템 프롬프트
        """
        self.client = client
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"Agent:{name}")

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트의 핵심 로직. 각 에이전트가 반드시 구현해야 합니다.

        Args:
            input_data: 에이전트별 입력 데이터 (키는 에이전트마다 상이)

        Returns:
            에이전트별 출력 데이터 (키는 에이전트마다 상이)
        """
        ...

    def _call_llm(self, user_message: str, temperature: float = 0.7) -> str:
        """
        OpenAI API를 호출하여 텍스트 응답을 반환합니다.

        Args:
            user_message: 사용자(에이전트 입력) 메시지
            temperature: 샘플링 온도 (기본 0.7)

        Returns:
            어시스턴트 응답 텍스트. 에러 시 빈 문자열.
        """
        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            if not response.choices or not response.choices[0].message.content:
                self.logger.warning("LLM 응답이 비어 있음")
                return ""
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error("LLM 호출 실패: %s", e, exc_info=True)
            return ""

    def _call_llm_json(self, user_message: str, temperature: float = 0.3) -> Dict[str, Any]:
        """
        OpenAI API를 JSON 모드로 호출하여 파싱된 딕셔너리를 반환합니다.

        Args:
            user_message: 사용자(에이전트 입력) 메시지
            temperature: 샘플링 온도 (기본 0.3)

        Returns:
            파싱된 JSON 객체. 파싱 실패 또는 에러 시 빈 dict.
        """
        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content if response.choices else None
            if not content:
                self.logger.warning("LLM JSON 응답이 비어 있음")
                return {}
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error("LLM JSON 파싱 실패: %s", e, exc_info=True)
            return {}
        except Exception as e:
            self.logger.error("LLM JSON 호출 실패: %s", e, exc_info=True)
            return {}

    def __repr__(self) -> str:
        return f"<{self.name} ({self.role})>"
