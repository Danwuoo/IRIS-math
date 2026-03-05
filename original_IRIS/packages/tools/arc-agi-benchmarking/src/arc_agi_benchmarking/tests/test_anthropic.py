import pytest
from unittest.mock import Mock, MagicMock, patch
from arc_agi_benchmarking.adapters.anthropic import AnthropicAdapter
from arc_agi_benchmarking.schemas import ModelConfig, ModelPricing
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def mock_model_config():
    """Provides a mock ModelConfig for Anthropic."""
    return ModelConfig(
        name="test-claude-model",
        model_name="claude-3-7-sonnet-20250219",
        provider="anthropic",
        pricing=ModelPricing(date="2025-03-12", input=3.0, output=15.0),
        kwargs={"max_tokens": 8192}
    )


@pytest.fixture
def adapter_instance(mock_model_config):
    """Creates an AnthropicAdapter instance with mocked config."""
    with patch.object(AnthropicAdapter, 'init_client') as mock_init:
        mock_init.return_value = Mock()
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        adapter.model_config = mock_model_config
        adapter.client = mock_init.return_value
        return adapter


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic message response."""
    mock_response = Mock()
    mock_response.id = "msg_test123"
    mock_response.model = "claude-3-7-sonnet-20250219"
    mock_response.role = "assistant"

    # Mock content blocks with valid JSON array response
    mock_content = Mock()
    mock_content.type = "text"
    mock_content.text = "[[1, 2], [3, 4]]"
    mock_response.content = [mock_content]

    # Mock usage
    mock_response.usage = Mock()
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 25

    return mock_response


class TestAnthropicStreaming:
    """Test streaming functionality for Anthropic adapter."""

    def test_streaming_enabled(self, adapter_instance, mock_anthropic_response):
        """Test that streaming is used when stream=True in config."""
        # Enable streaming
        adapter_instance.model_config.stream = True
        adapter_instance.model_config.kwargs = {'stream': True, 'max_tokens': 8192}

        # Mock the chat_completion_stream method
        with patch.object(adapter_instance, 'chat_completion_stream') as mock_stream:
            mock_stream.return_value = mock_anthropic_response

            messages = [{"role": "user", "content": "Hello"}]

            # Call make_prediction which should route to streaming
            attempt = adapter_instance.make_prediction("Hello")

            # Verify chat_completion_stream was called
            mock_stream.assert_called_once()
            assert attempt.answer == [[1, 2], [3, 4]]

    def test_streaming_disabled(self, adapter_instance, mock_anthropic_response):
        """Test that regular completion is used when stream=False."""
        # Disable streaming
        adapter_instance.model_config.kwargs = {'max_tokens': 8192}

        # Mock the chat_completion method
        with patch.object(adapter_instance, 'chat_completion') as mock_completion:
            mock_completion.return_value = mock_anthropic_response

            # Call make_prediction which should route to regular completion
            attempt = adapter_instance.make_prediction("Hello")

            # Verify chat_completion was called (not streaming)
            mock_completion.assert_called_once()
            assert attempt.answer == [[1, 2], [3, 4]]

    def test_chat_completion_stream_method(self, adapter_instance, mock_anthropic_response):
        """Test the chat_completion_stream method directly."""
        messages = [{"role": "user", "content": "Test message"}]

        # Mock the Anthropic client's messages.stream context manager
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value.get_final_message.return_value = mock_anthropic_response

        with patch.object(adapter_instance.client.messages, 'stream', return_value=mock_stream_context):
            result = adapter_instance.chat_completion_stream(messages)

            # Verify the result is the final message
            assert result == mock_anthropic_response
            assert result.id == "msg_test123"

            # Verify stream was called with correct parameters
            adapter_instance.client.messages.stream.assert_called_once_with(
                model="claude-3-7-sonnet-20250219",
                messages=messages,
                tools=[],
                max_tokens=8192
            )

    def test_streaming_with_tools(self, adapter_instance, mock_anthropic_response):
        """Test that streaming works with tools parameter."""
        messages = [{"role": "user", "content": "Test with tools"}]
        tools = [{"name": "test_tool", "description": "A test tool"}]

        # Mock the stream context
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__.return_value.get_final_message.return_value = mock_anthropic_response

        with patch.object(adapter_instance.client.messages, 'stream', return_value=mock_stream_context):
            result = adapter_instance.chat_completion_stream(messages, tools=tools)

            # Verify tools were passed correctly
            adapter_instance.client.messages.stream.assert_called_once_with(
                model="claude-3-7-sonnet-20250219",
                messages=messages,
                tools=tools,
                max_tokens=8192
            )
            assert result == mock_anthropic_response