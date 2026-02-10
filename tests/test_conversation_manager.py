"""
ConversationManager 테스트 모듈

이 모듈은 ConversationManager 클래스의 핵심 기능을 테스트합니다.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.conversation_manager import (
    ConversationManager,
    APIKeyNotFoundError,
)


class TestConversationManagerInitialization:
    """ConversationManager 초기화 테스트"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_init_with_default_system_message(self, mock_openai):
        """기본 시스템 메시지로 초기화 테스트"""
        # Mock OpenAI client 설정
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # ConversationManager 초기화
        manager = ConversationManager()
        
        # 검증
        assert manager is not None
        assert manager.message_count == 0
        assert manager.state == "idle"
        assert len(manager.messages) == 1  # 시스템 메시지 1개
        assert manager.messages[0]["role"] == "system"
        assert "리서치 어시스턴트" in manager.messages[0]["content"] or len(manager.messages[0]["content"]) > 0
        mock_openai.assert_called_once_with(api_key="test-api-key")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_init_with_custom_system_message(self, mock_openai):
        """커스텀 시스템 메시지로 초기화 테스트"""
        # Mock OpenAI client 설정
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        custom_message = "당신은 테스트 어시스턴트입니다."
        manager = ConversationManager(system_message=custom_message)
        
        # 검증
        assert manager.messages[0]["role"] == "system"
        assert manager.messages[0]["content"] == custom_message
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self):
        """API 키 없이 초기화 시도 테스트"""
        with pytest.raises(APIKeyNotFoundError) as exc_info:
            ConversationManager()
        
        assert "OPENAI_API_KEY" in str(exc_info.value)


class TestSystemMessage:
    """시스템 메시지 설정 테스트"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_default_system_message_content(self, mock_openai):
        """기본 시스템 메시지 내용 확인"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        system_message = manager.messages[0]["content"]
        assert len(system_message) > 0
        assert system_message is not None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_custom_system_message_preserved(self, mock_openai):
        """커스텀 시스템 메시지가 올바르게 저장되는지 확인"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        test_message = "커스텀 테스트 메시지"
        manager = ConversationManager(system_message=test_message)
        
        assert manager.get_messages()[0]["content"] == test_message
        assert manager.get_messages()[0]["role"] == "system"


class TestMessageManagement:
    """메시지 관리 테스트"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_get_messages_returns_copy(self, mock_openai):
        """get_messages가 복사본을 반환하는지 확인"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        messages1 = manager.get_messages()
        messages2 = manager.get_messages()
        
        # 다른 객체인지 확인 (복사본)
        assert messages1 is not messages2
        # 내용은 동일한지 확인
        assert messages1 == messages2
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_get_message_count_initial(self, mock_openai):
        """초기 메시지 카운트 확인"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        assert manager.get_message_count() == 0
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_clear_history(self, mock_openai):
        """대화 히스토리 초기화 테스트"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        # 초기 상태 확인
        assert len(manager.get_messages()) == 1  # 시스템 메시지
        
        # 히스토리 초기화
        manager.clear_history()
        
        # 시스템 메시지만 남아있는지 확인
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert manager.get_message_count() == 0
        assert manager.get_state() == "idle"


class TestStateDetermination:
    """상태 판단 테스트"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_determine_state_researching_keywords(self, mock_openai):
        """리서치 키워드 감지 테스트"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        # 리서치 키워드 테스트
        test_cases = [
            ("Python을 조사해줘", "researching"),
            ("이 주제를 분석해주세요", "researching"),
            ("리서치를 해주세요", "researching"),
            ("이것에 대해 알아봐줘", "researching"),
            ("찾아봐줘", "researching"),
            ("조사", "researching"),
        ]
        
        for user_input, expected_state in test_cases:
            state = manager.determine_state(user_input)
            assert state == expected_state, f"'{user_input}' should return '{expected_state}'"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_determine_state_responding(self, mock_openai):
        """일반 응답 상태 테스트"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        # 일반 응답 키워드 테스트
        test_cases = [
            "안녕하세요",
            "Python에 대해 알려주세요",
            "고마워요",
            "설명해줘",
        ]
        
        for user_input in test_cases:
            state = manager.determine_state(user_input)
            assert state == "responding", f"'{user_input}' should return 'responding'"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_determine_state_case_insensitive(self, mock_openai):
        """대소문자 구분 없이 키워드 감지 테스트"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        # 대소문자 혼합 테스트
        test_cases = [
            ("조사해줘", "researching"),
            ("조사해줘", "researching"),
            ("조사해줘", "researching"),
            ("리서치", "researching"),
            ("리서치", "researching"),
        ]
        
        for user_input, expected_state in test_cases:
            state = manager.determine_state(user_input)
            assert state == expected_state
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_get_state(self, mock_openai):
        """get_state 메서드 테스트"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        # 초기 상태는 idle
        assert manager.get_state() == "idle"
        
        # 상태 변경 후 확인
        manager.state = "researching"
        assert manager.get_state() == "researching"
        
        manager.state = "responding"
        assert manager.get_state() == "responding"


class TestChatMethod:
    """chat 메서드 테스트 (모킹 사용)"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_chat_adds_user_message(self, mock_openai):
        """chat 메서드가 사용자 메시지를 추가하는지 확인"""
        # Mock OpenAI client 및 응답 설정
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "테스트 응답"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        initial_count = len(manager.get_messages())
        
        # chat 메서드 호출
        response = manager.chat("테스트 입력")
        
        # 검증
        assert response == "테스트 응답"
        messages = manager.get_messages()
        assert len(messages) == initial_count + 2  # 사용자 메시지 + AI 응답
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"] == "테스트 입력"
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "테스트 응답"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_chat_updates_message_count(self, mock_openai):
        """chat 메서드가 메시지 카운트를 업데이트하는지 확인"""
        # Mock OpenAI client 및 응답 설정
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "응답"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        initial_count = manager.get_message_count()
        
        # chat 메서드 호출
        manager.chat("테스트")
        
        # 검증
        assert manager.get_message_count() == initial_count + 1
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_chat_updates_state(self, mock_openai):
        """chat 메서드가 상태를 업데이트하는지 확인"""
        # Mock OpenAI client 및 응답 설정
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "응답"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        # 리서치 키워드가 포함된 입력
        manager.chat("조사해줘")
        
        # 응답 후 상태는 idle로 변경되어야 함
        assert manager.get_state() == "idle"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    @patch("src.conversation_manager.OpenAI")
    def test_chat_empty_input_raises_error(self, mock_openai):
        """빈 입력 시 ValueError 발생 테스트"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        manager = ConversationManager()
        
        with pytest.raises(ValueError):
            manager.chat("")
        
        with pytest.raises(ValueError):
            manager.chat("   ")  # 공백만 있는 경우


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
