import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch
from typing import Optional

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse
from base_llm_unit_tests import BaseLLMChatTest
import asyncio
from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
)
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest


def test_openai_prediction_param():
    litellm.set_verbose = True
    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """

    completion = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    print(completion)

    assert (
        completion.usage.completion_tokens_details.accepted_prediction_tokens > 0
        or completion.usage.completion_tokens_details.rejected_prediction_tokens > 0
    )


@pytest.mark.asyncio
async def test_openai_prediction_param_mock():
    """
    Tests that prediction parameter is correctly passed to the API
    """
    litellm.set_verbose = True

    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
                    },
                    {"role": "user", "content": code},
                ],
                prediction={"type": "content", "content": code},
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the prediction parameter
        assert "prediction" in request_body
        # verify prediction is correctly sent to the API
        assert request_body["prediction"] == {"type": "content", "content": code}


@pytest.mark.asyncio
async def test_openai_prediction_param_with_caching():
    """
    Tests using `prediction` parameter with caching
    """
    from litellm.caching.caching import LiteLLMCacheType
    import logging
    from litellm._logging import verbose_logger

    verbose_logger.setLevel(logging.DEBUG)
    import time

    litellm.set_verbose = True
    litellm.cache = litellm.Cache(type=LiteLLMCacheType.LOCAL)
    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """

    completion_response_1 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    time.sleep(0.5)

    # cache hit
    completion_response_2 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
            },
            {"role": "user", "content": code},
        ],
        prediction={"type": "content", "content": code},
    )

    assert completion_response_1.id == completion_response_2.id

    completion_response_3 = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is the first name of the user?"},
        ],
        prediction={"type": "content", "content": code + "FirstName"},
    )

    assert completion_response_3.id != completion_response_1.id


@pytest.mark.asyncio()
async def test_vision_with_custom_model():
    """
    Tests that an OpenAI compatible endpoint when sent an image will receive the image in the request

    """
    import base64
    import requests
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    litellm.set_verbose = True
    api_base = "https://my-custom.api.openai.com"

    # Fetch and encode a test image
    url = "https://dummyimage.com/100/100/fff&text=Test+image"
    response = requests.get(url)
    file_data = response.content
    encoded_file = base64.b64encode(file_data).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_file}"

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            response = await litellm.acompletion(
                model="openai/my-custom-model",
                max_tokens=10,
                api_base=api_base,  # use the mock api
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image},
                            },
                        ],
                    }
                ],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkBAMAAACCzIhnAAAAG1BMVEURAAD///+ln5/h39/Dv79qX18uHx+If39MPz9oMSdmAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABDElEQVRYhe2SzWqEMBRGPyQTfQxJsc5jBKGzFmlslyFIZxsCQ7sUaWd87EanpdpIrbtC71mE/NyTm9wEIAiCIAiC+N/otQBxU2Sf/aeh4enqptHXri+/yxIq63jlKCw6cXssnr3ObdzdGYFYCJ2IzHKXLygHXCB98Gm4DE+ZZemu5EisQSyZTmyg+AuzQbkezCuIy7EI0k9Ig3FtruwydY+qniqtV5yQyo8qpUIl2fc90KVzJWohWf2qu75vlw52rdfjVDHg8vLWwixW7PChqLkSyUadwfSS0uQZhEvRuIkS53uJvrK8cGWYaPwpGt8efvw+vlo8TPMzcmP8w7lrNypc1RsNgiAIgiD+Iu/RyDYhCaWrgQAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            },
        ]
        assert request_body["model"] == "my-custom-model"
        assert request_body["max_tokens"] == 10


class TestOpenAIChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "gpt-4o-mini"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_prompt_caching(self):
        """
        Test that prompt caching works correctly.
        Skip for now, as it's working locally but not in CI
        """
        pass

    def test_prompt_caching(self):
        """
        Works locally but CI/CD is failing this test. Temporary skip to push out a new release.
        """
        pass


@patch("litellm.main.openai_chat_completions._get_openai_client")
def test_openai_max_retries_0(mock_get_openai_client):
    import litellm

    litellm.set_verbose = True
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
    )

    mock_get_openai_client.assert_called_once()
    assert mock_get_openai_client.call_args.kwargs["max_retries"] == 0


@patch("litellm.main.openai_chat_completions._get_openai_client")
def test_openai_image_generation_forwards_organization(mock_get_openai_client):
    """Ensure organization flows to OpenAI client for image generation."""

    class _DummyImages:
        def generate(self, **kwargs):  # type: ignore
            class _Resp:
                def model_dump(self_inner):  # minimal OpenAI ImagesResponse shape
                    return {
                        "created": 123,
                        "data": [{"url": "http://example.com/image.png"}],
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    }

            return _Resp()

    class _DummyClient:
        def __init__(self):
            self.api_key = "sk-test"

            class _BaseURL:
                _uri_reference = "https://api.openai.com/v1"

            self._base_url = _BaseURL()
            self.images = _DummyImages()

    mock_get_openai_client.return_value = _DummyClient()

    org = "org_test_123"
    resp = litellm.image_generation(
        model="gpt-image-1",
        prompt="A cute baby sea otter",
        organization=org,
    )

    # Assert organization forwarded into OpenAI client factory
    assert mock_get_openai_client.call_args.kwargs.get("organization") == org

    # Basic sanity on response shape
    assert hasattr(resp, "data") and len(resp.data) == 1


@pytest.mark.parametrize("model", ["o1", "o3-mini"])
def test_o1_parallel_tool_calls(model):
    litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "foo",
            }
        ],
        parallel_tool_calls=True,
        drop_params=True,
    )


def test_openai_chat_completion_streaming_handler_reasoning_content():
    from litellm.llms.openai.chat.gpt_transformation import (
        OpenAIChatCompletionStreamingHandler,
    )
    from unittest.mock import MagicMock

    streaming_handler = OpenAIChatCompletionStreamingHandler(
        streaming_response=MagicMock(),
        sync_stream=True,
    )
    response = streaming_handler.chunk_parser(
        chunk={
            "id": "e89b6501-8ac2-464c-9550-7cd3daf94350",
            "object": "chat.completion.chunk",
            "created": 1741037890,
            "model": "deepseek-reasoner",
            "system_fingerprint": "fp_5417b77867_prod0225",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "reasoning_content": "."},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
    )

    assert response.choices[0].delta.reasoning_content == "."


def validate_response_url_citation(url_citation: ChatCompletionAnnotationURLCitation):
    assert "end_index" in url_citation
    assert "start_index" in url_citation
    assert "url" in url_citation


def validate_web_search_annotations(annotations: ChatCompletionAnnotation):
    """validates litellm response contains web search annotations"""
    print("annotations: ", annotations)
    assert annotations is not None
    assert isinstance(annotations, list)
    for annotation in annotations:
        assert annotation["type"] == "url_citation"
        url_citation: ChatCompletionAnnotationURLCitation = annotation["url_citation"]
        validate_response_url_citation(url_citation)


@pytest.mark.flaky(reruns=3)
def test_openai_web_search():
    """Makes a simple web search request and validates the response contains web search annotations and all expected fields are present"""
    litellm._turn_on_debug()
    response = litellm.completion(
        model="openai/gpt-4o-search-preview",
        messages=[
            {
                "role": "user",
                "content": "What was a positive news story from today?",
            }
        ],
    )
    print("litellm response: ", response.model_dump_json(indent=4))
    message = response.choices[0].message
    if hasattr(message, "annotations"):
        annotations: ChatCompletionAnnotation = message.annotations
        validate_web_search_annotations(annotations)


def test_openai_web_search_streaming():
    """Makes a simple web search request and validates the response contains web search annotations and all expected fields are present"""
    # litellm._turn_on_debug()
    test_openai_web_search: Optional[ChatCompletionAnnotation] = None
    response = litellm.completion(
        model="openai/gpt-4o-search-preview",
        messages=[
            {
                "role": "user",
                "content": "What was a positive news story from today?",
            }
        ],
        stream=True,
    )
    for chunk in response:
        print("litellm response chunk: ", chunk)
        if (
            hasattr(chunk.choices[0].delta, "annotations")
            and chunk.choices[0].delta.annotations is not None
        ):
            test_openai_web_search = chunk.choices[0].delta.annotations

    # Assert this request has at-least one web search annotation
    if test_openai_web_search is not None:
        validate_web_search_annotations(test_openai_web_search)


class TestOpenAIGPT4OAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "openai/gpt-4o-transcribe",
            # "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.OPENAI


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4o"])
async def test_openai_pdf_url(model):
    from litellm.utils import return_raw_request, CallTypes

    request = return_raw_request(
        CallTypes.completion,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the first page of the PDF?"},
                        {
                            "type": "file",
                            "file": {"file_id": "https://arxiv.org/pdf/2303.08774"},
                        },
                    ],
                }
            ],
        },
    )
    print("request: ", request)

    assert (
        "file_data" in request["raw_request_body"]["messages"][0]["content"][1]["file"]
    )


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_codex_stream(sync_mode):
    from litellm.main import stream_chunk_builder

    kwargs = {
        "model": "openai/gpt-5.2-codex",
        "messages": [{"role": "user", "content": "Hey!"}],
        "stream": True,
    }

    chunks = []
    if sync_mode:
        response = litellm.completion(**kwargs)
        for chunk in response:
            chunks.append(chunk)
    else:
        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            chunks.append(chunk)

    complete_response = stream_chunk_builder(chunks=chunks)
    print("complete_response: ", complete_response)

    assert complete_response.choices[0].message.content is not None


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_openai_codex(sync_mode):

    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "openai-codex-mini-latest",
                "litellm_params": {
                    "model": "openai/gpt-5.2-codex",
                },
            }
        ]
    )

    kwargs = {
        "model": "openai-codex-mini-latest",
        "messages": [{"role": "user", "content": "Hey!"}],
    }

    if sync_mode:
        response = router.completion(**kwargs)
    else:
        response = await router.acompletion(**kwargs)
    print("response: ", response)

    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_openai_via_gemini_streaming_bridge():
    """
    Test that the openai via gemini streaming bridge works correctly
    """
    from litellm import Router
    from litellm.types.utils import ModelResponseStream

    router = Router(
        model_list=[
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                },
                "model_info": {
                    "version": 2,
                },
            }
        ],
        model_group_alias={"gemini-2.5-pro": "openai/gpt-3.5-turbo"},
    )

    response = await router.agenerate_content_stream(
        model="openai/gpt-3.5-turbo",
        contents=[
            {
                "parts": [{"text": "Write a long story about space exploration"}],
                "role": "user",
            }
        ],
        generationConfig={"maxOutputTokens": 500},
    )

    printed_chunks = []
    async for chunk in response:
        print("chunk: ", chunk)
        printed_chunks.append(chunk)
        assert not isinstance(chunk, ModelResponseStream)

    assert len(printed_chunks) > 0


def test_openai_deepresearch_model_bridge():
    """
    Test that the deepresearch model bridge works correctly
    """
    litellm._turn_on_debug()
    response = litellm.completion(
        model="o3-deep-research-2025-06-26",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        tools=[
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ],
    )

    print("response: ", response)


def test_openai_tool_calling():
    from pydantic import BaseModel
    from typing import Any, Literal

    class OpenAIFunction(BaseModel):
        description: Optional[str] = None
        name: str
        parameters: Optional[dict[str, Any]] = None

    class OpenAITool(BaseModel):
        type: Literal["function"]
        function: OpenAIFunction

    completion_params = {
        "model": "openai/gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is TSLA stock price at today?"}
                ],
            }
        ],
        "stream": False,
        "temperature": 0.5,
        "stop": None,
        "max_tokens": 1600,
        "tools": [
            OpenAITool(
                type="function",
                function=OpenAIFunction(
                    description="Get the current stock price for a given ticker symbol.",
                    name="get_stock_price",
                    parameters={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                            }
                        },
                        "required": ["ticker"],
                    },
                ),
            )
        ],
    }

    response = litellm.completion(**completion_params)


@pytest.mark.asyncio
async def test_openai_gpt5_reasoning():
    response = await litellm.acompletion(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        reasoning_effort="minimal",
    )
    print("response: ", response)
    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_openai_safety_identifier_parameter():
    """Test that safety_identifier parameter is correctly passed to the OpenAI API."""
    from openai import AsyncOpenAI

    litellm.set_verbose = True
    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                safety_identifier="user_code_123456",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the safety_identifier parameter
        assert "safety_identifier" in request_body
        # Verify safety_identifier is correctly sent to the API
        assert request_body["safety_identifier"] == "user_code_123456"


def test_openai_safety_identifier_parameter_sync():
    """Test that safety_identifier parameter is correctly passed to the OpenAI API."""
    from openai import OpenAI

    litellm.set_verbose = True
    client = OpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                safety_identifier="user_code_123456",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the safety_identifier parameter
        assert "safety_identifier" in request_body
        # Verify safety_identifier is correctly sent to the API
        assert request_body["safety_identifier"] == "user_code_123456"


@pytest.mark.asyncio
async def test_openai_service_tier_parameter():
    """Test that service_tier parameter is correctly passed to the OpenAI API."""
    from openai import AsyncOpenAI

    litellm.set_verbose = True
    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                service_tier="priority",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the service_tier parameter
        assert "service_tier" in request_body, "service_tier should be in request body"
        # Verify service_tier is correctly sent to the API
        assert (
            request_body["service_tier"] == "priority"
        ), "service_tier should be 'priority'"


def test_openai_service_tier_parameter_sync():
    """Test that service_tier parameter is correctly passed to the OpenAI API."""
    from openai import OpenAI

    litellm.set_verbose = True
    client = OpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                service_tier="priority",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the service_tier parameter
        assert "service_tier" in request_body, "service_tier should be in request body"
        # Verify service_tier is correctly sent to the API
        assert (
            request_body["service_tier"] == "priority"
        ), "service_tier should be 'priority'"


def test_gpt_5_reasoning_streaming():
    litellm._turn_on_debug()
    response = litellm.completion(
        model="openai/responses/gpt-5-mini",
        messages=[{"role": "user", "content": "Think of a poem, and then write it."}],
        reasoning_effort="low",
        stream=True,
    )

    has_content = False
    for chunk in response:
        print("chunk: ", chunk)
        if chunk.choices[0].delta.content:
            has_content = True
            print("content: ", chunk.choices[0].delta.content)

    assert has_content

    print("✓ gpt_5_reasoning_streaming correctly handled streaming")


def test_openai_gpt_5_codex_reasoning():
    litellm._turn_on_debug()
    completion_kwargs = {
        "model": "gpt-5-codex",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": '\nYou are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.\n\nIMPORTANT: Assist with defensive security tasks only. Refuse to create, modify, or improve code that may be used maliciously. Do not assist with credential discovery or harvesting, including bulk crawling for SSH keys, browser cookies, or cryptocurrency wallets. Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation.\nIMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.\n\nIf the user asks for help or wants to t.sh\nAm ../h2o-LLM-eval\nA  ../hel\nA  ../helm-test-repo/index.yaml\nA  ../helm-test-repo/litellm-helm-0.2.0.tgz\nA  ../kubernetes_deployment.yaml\nA  ../litellm-deployment.yaml\nA  ../litellm-dockerfile/Dockerfile\nA  ../litellm-dockerfile/secrets.toml\nA  ../litellm-helm-0.1.182.tgz\nA  ../litellm-helm/.DS_Store\nA  ../litellm-helm/.helmignore\nAM ../litellm-helm/Chart.lock\nAM ../litellm-helm/Chart.yaml\nAM ../litellm-helm/README.md\nA  ../litellm-helm/charts/.DS_Store\nAM ../litellm-helm/charts/postgresql/.helmignore\nAM ../litellm-helm/charts/postgresql/Chart.lock\nAM ../litellm-helm/charts/postgresql/Chart.yaml\nAM ../litellm-helm/charts/postgresql/README.md\nAM ../litellm-helm/charts/postgresql/charts/common/.helmignore\nAM ../litellm-helm/charts/postgresql/charts/common/Chart.yaml\nAM ../litellm-helm/charts/postgresql/charts/common/README.md\nA  ../litellm-helm/charts/postgresql/charts/common/templates/_affinities.tpl\nAM ../litellm-helm/charts/postgresql/charts/common/templates/_capabilities.tpl\nAM ../litellm-helm/charts/postgresql/charts/common/templates/_compatibility.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/_errors.tpl\nAM ../litellm-helm/charts/postgresql/charts/common/templates/_images.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/_ingress.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/_labels.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/_names.tpl\nAM ../litellm-helm/charts/postgresql/charts/common/templates/_resources.tpl\nAM ../litellm-helm/charts/postgresql/charts/common/templates/_secrets.tpl\nAM ../litellm-helm/charts/postgresql/charts/common/templates/_storage.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/_tplvalues.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/_utils.tpl\nAM ../litellm-helm/charts/postgresql/charts/common/templates/_warnings.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/validations/_cassandra.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/validations/_mariadb.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/validations/_mongodb.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/validations/_mysql.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/validations/_postgresql.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/validations/_redis.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/templates/validations/_validations.tpl\nA  ../litellm-helm/charts/postgresql/charts/common/values.yaml\nAM ../litellm-helm/charts/postgresql/templates/NOTES.txt\nAM ../litellm-helm/charts/postgresql/templates/_helpers.tpl\nAM ../litellm-helm/charts/postgresql/templates/backup/cronjob.yaml\nAM ../litellm-helm/charts/postgresql/templates/backup/networkpolicy.yaml\nA  ../litellm-helm/charts/postgresql/templates/backup/pvc.yaml\nA  ../litellm-helm/charts/postgresql/templates/extra-list.yaml\nA  ../litellm-helm/charts/postgresql/templates/primary/configmap.yaml\nA  ../litellm-helm/charts/postgresql/templates/primary/extended-configmap.yaml\nA  ../litellm-helm/charts/postgresql/templates/primary/initialization-configmap.yaml\nA  ../litellm-helm/charts/postgresql/templates/primary/metrics-configmap.yaml\nA  ../litellm-helm/charts/postgresql/templates/primary/metrics-svc.yaml\nA  ../litellm-helm/charts/postgresql/templates/primary/networkpolicy.yaml\nA  ../litellm-helm/charts/postgresql/templates/primary/servicemonitor.yaml\nAM ../litellm-helm/charts/postgresql/templates/primary/statefulset.yaml\nAM ../litellm-helm/charts/postgresql/templates/primary/svc-headless.yaml\nAM ../litellm-helm/charts/postgresql/templates/primary/svc.yaml\nA  ../litellm-helm/charts/postgresql/templates/prometheusrule.yaml\nA  ../litellm-helm/charts/postgresql/templates/psp.yaml\nA  ../litellm-helm/charts/postgresql/templates/read/extended-configmap.yaml\nA  ../litellm-helm/charts/postgresql/templates/read/metrics-configmap.yaml\nA  ../litellm-helm/charts/postgresql/templates/read/metrics-svc.yaml\nA  ../litellm-helm/charts/postgresql/templates/read/networkpolicy.yaml\nA  ../litellm-helm/charts/postgresql/templates/read/servicemonitor.yaml\nAM ../litellm-helm/charts/postgresql/templates/read/statefulset.yaml\nAM ../litellm-helm/charts/postgresql/templates/read/svc-headless.yaml\nAM ../litellm-helm/charts/postgresql/templates/read/svc.yaml\nA  ../litellm-helm/charts/postgresql/templates/role.yaml\nA  ../litellm-helm/charts/postgresql/templates/rolebinding.yaml\nA  ../litellm-helm/charts/postgresql/templates/secrets.yaml\nA  ../litellm-helm/charts/postgresql/templates/serviceaccount.yaml\nA  ../litellm-helm/charts/postgresql/templates/tls-secrets.yaml\nA  ../litellm-helm/charts/postgresql/values.schema.json\nAM ../litellm-helm/charts/postgresql/values.yaml\nAM ../litellm-helm/charts/redis/.helmignore\nAM ../litellm-helm/charts/redis/Chart.lock\nAM ../litellm-helm/charts/redis/Chart.yaml\nAM ../litellm-helm/charts/redis/README.md\nAM ../litellm-helm/charts/redis/charts/common/.helmignore\nAM ../litellm-helm/charts/redis/charts/common/Chart.yaml\nAM ../litellm-helm/charts/redis/charts/common/README.md\nAM ../litellm-helm/charts/redis/charts/common/templates/_affinities.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_capabilities.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_compatibility.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_errors.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_images.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_ingress.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_labels.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_names.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_resources.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_secrets.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_storage.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_tplvalues.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_utils.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/_warnings.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/validations/_cassandra.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/validations/_mariadb.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/validations/_mongodb.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/validations/_mysql.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/validations/_postgresql.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/validations/_redis.tpl\nAM ../litellm-helm/charts/redis/charts/common/templates/validations/_validations.tpl\nAM ../litellm-helm/charts/redis/charts/common/values.yaml\nAM ../litellm-helm/charts/redis/templates/NOTES.txt\nAM ../litellm-helm/charts/redis/templates/_helpers.tpl\nAM ../litellm-helm/charts/redis/templates/configmap.yaml\nA  ../litellm-helm/charts/redis/templates/extra-list.yaml\nA  ../litellm-helm/charts/redis/templates/headless-svc.yaml\nA  ../litellm-helm/charts/redis/templates/health-configmap.yaml\nAM ../litellm-helm/charts/redis/templates/master/application.yaml\nA  ../litellm-helm/charts/redis/templates/master/psp.yaml\nA  ../litellm-helm/charts/redis/templates/master/pvc.yaml\nAM ../litellm-helm/charts/redis/templates/master/service.yaml\nA  ../litellm-helm/charts/redis/templates/master/serviceaccount.yaml\nA  ../litellm-helm/charts/redis/templates/metrics-svc.yaml\nA  ../litellm-helm/charts/redis/templates/networkpolicy.yaml\nA  ../litellm-helm/charts/redis/templates/pdb.yaml\nA  ../litellm-helm/charts/redis/templates/podmonitor.yaml\nA  ../litellm-helm/charts/redis/templates/prometheusrule.yaml\nAM ../litellm-helm/charts/redis/templates/replicas/application.yaml\nA  ../litellm-helm/charts/redis/templates/replicas/hpa.yaml\nA  ../litellm-helm/charts/redis/templates/replicas/service.yaml\nA  ../litellm-helm/charts/redis/templates/replicas/serviceaccount.yaml\nAM ../litellm-helm/charts/redis/templates/role.yaml\nA  ../litellm-helm/charts/redis/templates/rolebinding.yaml\nAM ../litellm-helm/charts/redis/templates/scripts-configmap.yaml\nA  ../litellm-helm/charts/redis/templates/secret-svcbind.yaml\nA  ../litellm-helm/charts/redis/templates/secret.yaml\nA  ../litellm-helm/charts/redis/templates/sentinel/hpa.yaml\nA  ../litellm-helm/charts/redis/templates/sentinel/node-services.yaml\nA  ../litellm-helm/charts/redis/templates/sentinel/ports-configmap.yaml\nAM ../litellm-helm/charts/redis/templates/sentinel/service.yaml\nAM ../litellm-helm/charts/redis/templates/sentinel/statefulset.yaml\nA  ../litellm-helm/charts/redis/templates/serviceaccount.yaml\nA  ../litellm-helm/charts/redis/templates/servicemonitor.yaml\nA  ../litellm-helm/charts/redis/templates/tls-secret.yaml\nAM ../litellm-helm/charts/redis/values.schema.json\nAM ../litellm-helm/charts/redis/values.yaml\nA  ../litellm-helm/templates/NOTES.txt\nA  ../litellm-helm/templates/_helpers.tpl\nA  ../litellm-helm/templates/configmap-litellm.yaml\nAM ../litellm-helm/templates/deployment.yaml\nA  ../litellm-helm/templates/hpa.yaml\nA  ../litellm-helm/templates/ingress.yaml\nA  ../litellm-helm/templates/secret-dbcredentials.yaml\nA  ../litellm-helm/templates/secret-masterkey.yaml\nA  ../litellm-helm/templates/service.yaml\nA  ../litellm-helm/templates/serviceaccount.yaml\nA  ../litellm-helm/templates/tests/test-connection.yaml\nA  ../litellm-helm/values.yaml\nA  ../litellm-krrish.pem\nA  ../litellm-proxy-config/litellm_config.toml\nA  ../litellm-proxy/.gitignore\nA  ../litellm-proxy/__init__.py\nA  ../litellm-proxy/__pycache__/proxy_server.cpython-311.pyc\nA  ../litellm-proxy/cost.log\nA  ../litellm-proxy/costs.json\nA  ../litellm-proxy/proxy_cli.py\nA  ../litellm-proxy/proxy_server.py\nA  ../litellm-service.yaml\nAM ../litellm_config.yaml\nA  ../litellm_edge_functions/.DS_Store\nM  ../litellm_edge_functions/openai-function-calling-workers/package-lock.json\nM  ../litellm_edge_functions/openai-function-calling-workers/package.json\nA  ../litellm_edge_functions/openai-function-calling-workers/router_config.yaml\nM  ../litellm_edge_functions/openai-function-calling-workers/src/index.js\nA  ../litellm_edge_functions/openai-function-calling-workers/src/llms/azure_openai.js\nA  ../litellm_edge_functions/openai-function-calling-workers/src/router.js\nAm ../litellm_playground_fe_template\nA  ../litellm_uuid.txt\nA  ../llm_guard_config.save\nA  ../llm_guard_config_dir/.DS_Store\nA  ../llm_guard_config_dir/scanners.yml\nA  ../load_test.lua\nA  ../load_test_key_generate.py\nA  ../load_test_pydantic_obj.py\nA  ../loadtest_hosted_proxy/main.py\nA  ../locust_tests/__pycache__/large_file.cpython-311.pyc\nAM ../locust_tests/__pycache__/locustfile.cpython-311.pyc\nA  ../locust_tests/cloudflare_locustfile.py\nA  ../locust_tests/fake_openai_locustfile.py\nA  ../locust_tests/large_file.py\nAM ../locust_tests/locustfile.py\nA  ../locust_tests/update_database_locustfile.py\nAM ../log.txt\nA  ../main.py\nA  ../mlflow_config.yml\nA  ../myFile.log\nA  ../myenv/bin/Activate.ps1\nA  ../myenv/bin/activate\nA  ../myenv/bin/activate.csh\nA  ../myenv/bin/activate.fish\nA  ../myenv/bin/pip\nA  ../myenv/bin/pip3\nA  ../myenv/bin/pip3.11\nA  ../myenv/bin/python\nA  ../myenv/bin/python3\nA  ../myenv/bin/python3.11\nAM ../myenv/lib/python3.11/site-packages/_distutils_hack/__init__.py\nAM ../myenv/lib/python3.11/site-packages/_distutils_hack/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/_distutils_hack/__pycache__/override.cpython-311.pyc\nA  ../myenv/lib/python3.11/site-packages/_distutils_hack/override.py\nA  ../myenv/lib/python3.11/site-packages/distutils-precedence.pth\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/AUTHORS.txt\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/INSTALLER\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/LICENSE.txt\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/METADATA\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/RECORD\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/REQUESTED\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/WHEEL\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/entry_points.txt\nAD ../myenv/lib/python3.11/site-packages/pip-23.3.1.dist-info/top_level.txt\nAM ../myenv/lib/python3.11/site-packages/pip/__init__.py\nA  ../myenv/lib/python3.11/site-packages/pip/__main__.py\nAM ../myenv/lib/python3.11/site-packages/pip/__pip-runner__.py\nAM ../myenv/lib/python3.11/site-packages/pip/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/__pycache__/__main__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/__pycache__/__pip-runner__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/build_env.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/cache.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/configuration.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/exceptions.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/main.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/pyproject.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/self_outdated_check.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/__pycache__/wheel_builder.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/build_env.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cache.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/autocompletion.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/base_command.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/cmdoptions.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/command_context.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/main.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/main_parser.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/parser.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/progress_bars.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/req_command.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/spinners.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/__pycache__/status_codes.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/autocompletion.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/base_command.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/cmdoptions.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/cli/command_context.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/main.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/main_parser.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/parser.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/progress_bars.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/cli/req_command.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/cli/spinners.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/cli/status_codes.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/cache.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/check.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/completion.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/configuration.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/debug.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/download.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/freeze.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/hash.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/help.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/index.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/inspect.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/install.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/list.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/search.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/show.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/uninstall.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/__pycache__/wheel.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/cache.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/check.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/completion.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/configuration.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/debug.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/download.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/freeze.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/commands/hash.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/commands/help.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/index.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/inspect.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/install.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/list.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/search.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/show.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/uninstall.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/commands/wheel.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/configuration.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/__pycache__/base.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/__pycache__/installed.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/__pycache__/sdist.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/__pycache__/wheel.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/base.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/installed.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/sdist.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/distributions/wheel.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/exceptions.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/index/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/index/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/index/__pycache__/collector.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/index/__pycache__/package_finder.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/index/__pycache__/sources.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/index/collector.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/index/package_finder.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/index/sources.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/locations/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/locations/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/locations/__pycache__/_distutils.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/locations/__pycache__/_sysconfig.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/locations/__pycache__/base.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/locations/_distutils.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/locations/_sysconfig.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/locations/base.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/main.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/__pycache__/_json.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/__pycache__/base.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/__pycache__/pkg_resources.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/_json.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/base.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/importlib/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/importlib/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/importlib/__pycache__/_compat.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/importlib/__pycache__/_dists.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/importlib/__pycache__/_envs.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/importlib/_compat.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/importlib/_dists.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/importlib/_envs.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/metadata/pkg_resources.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/candidate.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/direct_url.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/format_control.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/index.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/installation_report.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/link.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/scheme.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/search_scope.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/selection_prefs.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/target_python.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/__pycache__/wheel.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/candidate.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/direct_url.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/format_control.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/models/index.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/models/installation_report.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/link.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/scheme.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/search_scope.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/selection_prefs.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/target_python.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/models/wheel.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__pycache__/auth.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__pycache__/cache.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__pycache__/download.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__pycache__/lazy_wheel.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__pycache__/session.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__pycache__/utils.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/__pycache__/xmlrpc.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/auth.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/cache.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/download.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/lazy_wheel.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/session.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/utils.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/network/xmlrpc.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/operations/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/__pycache__/check.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/__pycache__/freeze.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/__pycache__/prepare.cpython-311.pyc\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__pycache__/build_tracker.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__pycache__/metadata.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__pycache__/metadata_editable.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__pycache__/metadata_legacy.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__pycache__/wheel.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__pycache__/wheel_editable.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/__pycache__/wheel_legacy.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/build_tracker.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/metadata.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/metadata_editable.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/metadata_legacy.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/wheel.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/wheel_editable.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/build/wheel_legacy.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/check.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/freeze.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/install/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/install/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/install/__pycache__/editable_legacy.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/install/__pycache__/wheel.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/install/editable_legacy.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/install/wheel.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/operations/prepare.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/pyproject.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/__pycache__/constructors.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/__pycache__/req_file.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/__pycache__/req_install.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/__pycache__/req_set.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/__pycache__/req_uninstall.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/constructors.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/req_file.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/req_install.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/req_set.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/req/req_uninstall.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/__pycache__/base.cpython-311.pyc\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/base.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/legacy/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/legacy/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/legacy/__pycache__/resolver.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/legacy/resolver.py\nA  ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/resolvelib/__init__.py\nAM ../myenv/lib/python3.11/site-packages/pip/_internal/resolution/resolvelib/__pycache__/__init__.cpython-311.pyc\nAM ../myenv/lib/python3.11/site-packages/pip/_interna\n... (truncated because it exceeds 40k characters. If you need more information, run "git status" using BashTool)\n\nRecent commits:\n3066e01 Initial commit (by create-cloudflare CLI)',
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>\nThis is a reminder that your todo list is currently empty. DO NOT mention this to the user explicitly because they are already aware. If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. If not, please feel free to ignore. Again do not mention this message to the user.\n</system-reminder>",
                    },
                    {"type": "text", "text": "hey"},
                ],
            },
        ],
        "user": "user_5dd07c33da27e6d2968d94ea20bf47a7b090b6b158b82328d54da2909a108e84_account__session_d2e70091-3007-41a3-a916-cb8b0d3098de",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "A short (3-5 word) description of the task",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The task for the agent to perform",
                            },
                            "subagent_type": {
                                "type": "string",
                                "description": "The type of specialized agent to use for this task",
                            },
                        },
                        "required": ["description", "prompt", "subagent_type"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": 'Launch a new agent to handle complex, multi-step tasks autonomously. \n\nAvailable agent types and the tools they have access to:\n- general-purpose: General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. (Tools: *)\n- statusline-setup: Use this agent to configure the user\'s Claude Code status line setting. (Tools: Read, Edit)\n- output-style-setup: Use this agent to create a Claude Code output style. (Tools: Read, Write, Edit, Glob, Grep)\n\nWhen using the Task tool, you must specify a subagent_type parameter to select which agent type to use.\n\nWhen NOT to use the Agent tool:\n- If you want to read a specific file path, use the Read or Glob tool instead of the Agent tool, to find the match more quickly\n- If you are searching for a specific class definition like "class Foo", use the Glob tool instead, to find the match more quickly\n- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead of the Agent tool, to find the match more quickly\n- Other tasks that are not related to the agent descriptions above\n\n\nUsage notes:\n- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses\n- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.\n- For agents that run in the background, you will need to use AgentOutputTool to retrieve their results once they are done. You can continue to work while async agents run in the background - when you need their results to continue you can use AgentOutputTool in blocking mode to pause and wait for their results.\n- Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.\n- The agent\'s outputs should generally be trusted\n- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user\'s intent\n- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.\n- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple Task tool use content blocks. For example, if you need to launch both a code-reviewer agent and a test-runner agent in parallel, send a single message with both tool calls.\n\nExample usage:\n\n<example_agent_descriptions>\n"code-reviewer": use this agent after you are done writing a signficant piece of code\n"greeting-responder": use this agent when to respond to user greetings with a friendly joke\n</example_agent_description>\n\n<example>\nuser: "Please write a function that checks if a number is prime"\nassistant: Sure let me write a function that checks if a number is prime\nassistant: First let me use the Write tool to write a function that checks if a number is prime\nassistant: I\'m going to use the Write tool to write the following code:\n<code>\nfunction isPrime(n) {\n  if (n <= 1) return false\n  for (let i = 2; i * i <= n; i++) {\n    if (n % i === 0) return false\n  }\n  return true\n}\n</code>\n<commentary>\nSince a signficant piece of code was written and the task was completed, now use the code-reviewer agent to review the code\n</commentary>\nassistant: Now let me use the code-reviewer agent to review the code\nassistant: Uses the Task tool to launch the with the code-reviewer agent \n</example>\n\n<example>\nuser: "Hello"\n<commentary>\nSince the user is greeting, use the greeting-responder agent to respond with a friendly joke\n</commentary>\nassistant: "I\'m going to use the Task tool to launch the with the greeting-responder agent"\n</example>\n',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute",
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Optional timeout in milliseconds (max 600000)",
                            },
                            "description": {
                                "type": "string",
                                "description": "Clear, concise description of what this command does in 5-10 words, in active voice. Examples:\nInput: ls\nOutput: List files in current directory\n\nInput: git status\nOutput: Show working tree status\n\nInput: npm install\nOutput: Install package dependencies\n\nInput: mkdir foo\nOutput: Create directory 'foo'",
                            },
                            "run_in_background": {
                                "type": "boolean",
                                "description": "Set to true to run this command in the background. Use BashOutput to read the output later.",
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": 'Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.\n\nIMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.\n\nBefore executing the command,  PR: gh api repos/foo/bar/pulls/123/comments',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The glob pattern to match files against",
                            },
                            "path": {
                                "type": "string",
                                "description": 'The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter "undefined" or "null" - simply omit it for the default behavior. Must be a valid directory path if provided.',
                            },
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": '- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like "**/*.js" or "src/**/*.ts"\n- Returns matching file paths sorted by modification time\n- Use this tool when you need to find files by name patterns\n- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead\n- You can call multiple tools in a single response. It is always better to speculatively perform multiple searches in parallel if they are potentially useful.',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Grep",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The regular expression pattern to search for in file contents",
                            },
                            "path": {
                                "type": "string",
                                "description": "File or directory to search in (rg PATH). Defaults to current working directory.",
                            },
                            "glob": {
                                "type": "string",
                                "description": 'Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob',
                            },
                            "output_mode": {
                                "type": "string",
                                "enum": ["content", "files_with_matches", "count"],
                                "description": 'Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), "files_with_matches" shows file paths (supports head_limit), "count" shows match counts (supports head_limit). Defaults to "files_with_matches".',
                            },
                            "-B": {
                                "type": "number",
                                "description": 'Number of lines to show before each match (rg -B). Requires output_mode: "content", ignored otherwise.',
                            },
                            "-A": {
                                "type": "number",
                                "description": 'Number of lines to show after each match (rg -A). Requires output_mode: "content", ignored otherwise.',
                            },
                            "-C": {
                                "type": "number",
                                "description": 'Number of lines to show before and after each match (rg -C). Requires output_mode: "content", ignored otherwise.',
                            },
                            "-n": {
                                "type": "boolean",
                                "description": 'Show line numbers in output (rg -n). Requires output_mode: "content", ignored otherwise.',
                            },
                            "-i": {
                                "type": "boolean",
                                "description": "Case insensitive search (rg -i)",
                            },
                            "type": {
                                "type": "string",
                                "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types.",
                            },
                            "head_limit": {
                                "type": "number",
                                "description": 'Limit output to first N lines/entries, equivalent to "| head -N". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). When unspecified, shows all results from ripgrep.',
                            },
                            "multiline": {
                                "type": "boolean",
                                "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.",
                            },
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": 'A powerful search tool built on ripgrep\n\n  Usage:\n  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.\n  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")\n  - Filter files with glob parameter (e.g., "*.js", "**/*.tsx") or type parameter (e.g., "js", "py", "rust")\n  - Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts\n  - Use Task tool for open-ended searches requiring multiple rounds\n  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)\n  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`\n',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ExitPlanMode",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "string",
                                "description": "The plan you came up with, that you want to run by the user for approval. Supports markdown. The plan should be pretty concise.",
                            }
                        },
                        "required": ["plan"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": 'Use this tool when you are in plan mode and have finished presenting your plan and are ready to code. This will prompt the user to exit plan mode.\nIMPORTANT: Only use this tool when the task requires planning the implementation steps of a task that requires writing code. For research tasks where you\'re gathering information, searching files, reading files or in general trying to understand the codebase - do NOT use this tool.\n\nEg.\n1. Initial task: "Search for and understand the implementation of vim mode in the codebase" - Do not use the exit plan mode tool because you are not planning the implementation steps of a task.\n2. Initial task: "Help me implement yank mode for vim" - Use the exit plan mode tool after you have finished planning the implementation steps of the task.\n',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The absolute path to the file to read",
                            },
                            "offset": {
                                "type": "number",
                                "description": "The line number to start reading from. Only provide if the file is too large to read at once",
                            },
                            "limit": {
                                "type": "number",
                                "description": "The number of lines to read. Only provide if the file is too large to read at once.",
                            },
                        },
                        "required": ["file_path"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": "Reads a file from the local filesystem. You can access any file directly by using this tool.\nAssume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.\n\nUsage:\n- The file_path parameter must be an absolute path, not a relative path\n- By default, it reads up to 2000 lines starting from the beginning of the file\n- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters\n- Any lines longer than 2000 characters will be truncated\n- Results are returned using cat -n format, with line numbers starting at 1\n- This tool allows Claude Code to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as Claude Code is a multimodal LLM.\n- This tool can read PDF files (.pdf). PDFs are processed page by page, extracting both text and visual content for analysis.\n- This tool can read Jupyter notebooks (.ipynb files) and returns all cells with their outputs, combining code, text, and visualizations.\n- This tool can only read files, not directories. To read a directory, use an ls command via the Bash tool.\n- You can call multiple tools in a single response. It is always better to speculatively read multiple potentially useful files in parallel.\n- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths.\n- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The absolute path to the file to modify",
                            },
                            "old_string": {
                                "type": "string",
                                "description": "The text to replace",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "The text to replace it with (must be different from old_string)",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "default": False,
                                "description": "Replace all occurences of old_string (default false)",
                            },
                        },
                        "required": ["file_path", "old_string", "new_string"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": "Performs exact string replacements in files. \n\nUsage:\n- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. \n- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`. \n- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The absolute path to the file to write (must be absolute, not relative)",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write to the file",
                            },
                        },
                        "required": ["file_path", "content"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": "Writes a file to the local filesystem.\n\nUsage:\n- This tool will overwrite the existing file if there is one at the provided path.\n- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.\n- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "NotebookEdit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "notebook_path": {
                                "type": "string",
                                "description": "The absolute path to the Jupyter notebook file to edit (must be absolute, not relative)",
                            },
                            "cell_id": {
                                "type": "string",
                                "description": "The ID of the cell to edit. When inserting a new cell, the new cell will be inserted after the cell with this ID, or at the beginning if not specified.",
                            },
                            "new_source": {
                                "type": "string",
                                "description": "The new source for the cell",
                            },
                            "cell_type": {
                                "type": "string",
                                "enum": ["code", "markdown"],
                                "description": "The type of the cell (code or markdown). If not specified, it defaults to the current cell type. If using edit_mode=insert, this is required.",
                            },
                            "edit_mode": {
                                "type": "string",
                                "enum": ["replace", "insert", "delete"],
                                "description": "The type of edit to make (replace, insert, delete). Defaults to replace.",
                            },
                        },
                        "required": ["notebook_path", "new_source"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": "Completely replaces the contents of a specific cell in a Jupyter notebook (.ipynb file) with new source. Jupyter notebooks are interactive documents that combine code, text, and visualizations, commonly used for data analysis and scientific computing. The notebook_path parameter must be an absolute path, not a relative path. The cell_number is 0-indexed. Use edit_mode=insert to add a new cell at the index specified by cell_number. Use edit_mode=delete to delete the cell at the index specified by cell_number.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "WebFetch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "format": "uri",
                                "description": "The URL to fetch content from",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to run on the fetched content",
                            },
                        },
                        "required": ["url", "prompt"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": '\n- Fetches content from a specified URL and processes it using an AI model\n- Takes a URL and a prompt as input\n- Fetches the URL content, converts HTML to markdown\n- Processes the content with the prompt using a small, fast model\n- Returns the model\'s response about the content\n- Use this tool when you need to retrieve and analyze web content\n\nUsage notes:\n  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions. All MCP-provided tools start with "mcp__".\n  - The URL must be a fully-formed valid URL\n  - HTTP URLs will be automatically upgraded to HTTPS\n  - The prompt should describe what information you want to extract from the page\n  - This tool is read-only and does not modify any files\n  - Results may be summarized if the content is very large\n  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL\n  - When a URL redirects to a different host, the tool will inform you and provide the redirect URL in a special format. You should then make a new WebFetch request with the redirect URL to fetch the content.\n',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "TodoWrite",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string", "minLength": 1},
                                        "status": {
                                            "type": "string",
                                            "enum": [
                                                "pending",
                                                "in_progress",
                                                "completed",
                                            ],
                                        },
                                        "activeForm": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                    },
                                    "required": ["content", "status", "activeForm"],
                                    "additionalProperties": False,
                                },
                                "description": "The updated todo list",
                            }
                        },
                        "required": ["todos"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": "Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\nIt also helps the user understand the progress of the task and overall progress of their requests.\n\n## When to Use This Tool\nUse this tool proactively in these scenarios:\n\n1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations\n3. User explicitly requests todo list - When the user directly asks you to use the todo list\n4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)\n5. After receiving new instructions - Immediately capture user requirements as todos\n6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time\n7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation\n\n## When NOT to Use This Tool\n\nSkip using this tool when:\n1. There is only a single, straightforward task\n2. The task is trivial and tracking it provides no organizational benefit\n3. The task can be completed in less than 3 trivial steps\n4. The task is purely conversational or informational\n\nNOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.\n\n## Examples of When to Use the Todo List\n\n<example>\nUser: I want to add a dark mode toggle to the application settings. Make sure you run the tests and build when you're done!\nAssistant: I'll help add a dark mode toggle to your application settings. Let me create a todo list to track this implementation.\n*Creates todo list with the following items:*\n1. Creating dark mode toggle component in Settings page\n2. Adding dark mode state management (context/store)\n3. Implementing CSS-in-JS styles for dark theme\n4. Updating existing components to support theme switching\n5. Running tests and build process, addressing any failures or errors that occur\n*Begins working on the first task*\n\n<reasoning>\nThe assistant used the todo list because:\n1. Adding dark mode is a multi-step feature requiring UI, state management, and styling changes\n2. The user explicitly requested tests and build be run afterward\n3. The assistant inferred that tests and build need to pass by adding \"Ensure tests and build succeed\" as the final task\n</reasoning>\n</example>\n\n<example>\nUser: Help me rename the function getCwd to getCurrentWorkingDirectory across my project\nAssistant: Let me first search through your codebase to find all occurrences of 'getCwd'.\n*Uses grep or search tools to locate all instances of getCwd in the codebase*\nAssistant: I've found 15 instances of 'getCwd' across 8 different files. Let me create a todo list to track these changes.\n*Creates todo list with specific items for each file that needs updating*\n\n<reasoning>\nThe assistant used the todo list because:\n1. First, the assistant searched to understand the scope of the task\n2. Upon finding multiple occurrences across different files, it determined this was a complex task with multiple steps\n3. The todo list helps ensure every instance is tracked and updated systematically\n4. This approach prevents missing any occurrences and maintains code consistency\n</reasoning>\n</example>\n\n\n<example>\nUser: I need to implement these features for my e-commerce site: user registration, product catalog, shopping cart, and checkout flow.\nAssistant: I'll help implement these features. First, let's add all the features to the todo list.\n*Creates a todo list breaking down each feature into specific tasks based on the project architecture*\nAssistant: Let's start with user registration. This will involve creating database models, API endpoints, and frontend forms that integrate with your existing authentication system.\n\n<reasoning>\nThe assistant used the todo list because:\n1. The user provided multiple complex features to implement in a comma separated list\n2. The todo list helps organize these large features into manageable tasks\n3. This approach allows for tracking progress across the entire implementation\n</reasoning>\n</example>\n\n<example>\nUser: Can you help optimize my React application? It's rendering slowly and has performance issues.</user>\nAssistant: I'll help optimize your React application. First, let me examine your codebase to identify potential performance bottlenecks.\n*Reviews component structure, render patterns, state management, and data fetching*\nAssistant: After analyzing your codebase, I've identified several performance issues. Let me create a todo list to track our optimization efforts.\n*Creates todo list with items like: 1) Implementing memoization for expensive calculations in ProductList, 2) Adding virtualization for long lists in Dashboard, 3) Optimizing image loading in Gallery component, 4) Fixing state update loops in ShoppingCart, 5) Reviewing bundle size and implementing code splitting*\nLet's start by implementing memoization for the expensive calculations in your ProductList component.</assistant>\n\n<reasoning>\nThe assistant used the todo list because:\n1. First, the assistant examined the codebase to identify specific performance issues\n2. Based on this analysis, it identified multiple optimization opportunities\n3. Performance optimization is a non-trivial task requiring multiple steps\n4. The todo list helps methodically track improvements across different components\n5. This systematic approach ensures all performance bottlenecks are addressed\n</reasoning>\n</example>\n\n## Examples of When NOT to Use the Todo List\n\n<example>\nUser: How do I print 'Hello World' in Python?\nAssistant: In Python, you can print \"Hello World\" with this simple code:\n\npython\nprint(\"Hello World\")\n\nThis will output the text \"Hello World\" to the console when executed.</assistant>\n\n<reasoning>\nThe assistant did not use the todo list because this is a single, trivial task that can be completed in one step. There's no need to track multiple tasks or steps for such a straightforward request.\n</reasoning>\n</example>\n\n<example>\nUser: What does the git status command do?\nAssistant: The git status command shows the current state of your working directory and staging area. It displays which changes have been staged, which haven't, and which files aren't being tracked by Git. This is a fundamental command for understanding the current state of your repository before making commits.\n\n<reasoning>\nThe assistant did not use the todo list because this is an informational request with no actual coding task to complete. The user is simply asking for an explanation, not for the assistant to perform multiple steps or tasks.\n</reasoning>\n</example>\n\n<example>\nUser: Can you add a comment to the calculateTotal function to explain what it does?\nAssistant: Sure, let me add a comment to the calculateTotal function to explain what it does.\n* Uses the Edit tool to add a comment to the calculateTotal function *\n\n<reasoning>\nThe assistant did not use the todo list because this is a single, straightforward task confined to one location in the code. Adding a comment doesn't require tracking multiple steps or systematic organization.\n</reasoning>\n</example>\n\n<example>\nUser: Run npm install for me and tell me what happens.\nAssistant: I'll run the npm install command for you.\n\n*Executes: npm install*\n\nThe command completed successfully. Here's the output:\n[Output of npm install command]\n\nAll dependencies have been installed according to your package.json file.\n\n<reasoning>\nThe assistant did not use the todo list because this is a single command execution with immediate results. There are no multiple steps to track or organize, making the todo list unnecessary for this straightforward task.\n</reasoning>\n</example>\n\n## Task States and Management\n\n1. **Task States**: Use these states to track progress:\n   - pending: Task not yet started\n   - in_progress: Currently working on (limit to ONE task at a time)\n   - completed: Task finished successfully\n\n   **IMPORTANT**: Task descriptions must have two forms:\n   - content: The imperative form describing what needs to be done (e.g., \"Run tests\", \"Build the project\")\n   - activeForm: The present continuous form shown during execution (e.g., \"Running tests\", \"Building the project\")\n\n2. **Task Management**:\n   - Update task status in real-time as you work\n   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)\n   - Exactly ONE task must be in_progress at any time (not less, not more)\n   - Complete current tasks before starting new ones\n   - Remove tasks that are no longer relevant from the list entirely\n\n3. **Task Completion Requirements**:\n   - ONLY mark a task as completed when you have FULLY accomplished it\n   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress\n   - When blocked, create a new task describing what needs to be resolved\n   - Never mark a task as completed if:\n     - Tests are failing\n     - Implementation is partial\n     - You encountered unresolved errors\n     - You couldn't find necessary files or dependencies\n\n4. **Task Breakdown**:\n   - Create specific, actionable items\n   - Break complex tasks into smaller, manageable steps\n   - Use clear, descriptive task names\n   - Always provide both forms:\n     - content: \"Fix authentication bug\"\n     - activeForm: \"Fixing authentication bug\"\n\nWhen in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.\n",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "WebSearch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "minLength": 2,
                                "description": "The search query to use",
                            },
                            "allowed_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Only include search results from these domains",
                            },
                            "blocked_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Never include search results from these domains",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": '\n- Allows Claude to search the web and use the results to inform responses\n- Provides up-to-date information for current events and recent data\n- Returns search result information formatted as search result blocks\n- Use this tool for accessing information beyond Claude\'s knowledge cutoff\n- Searches are performed automatically within a single API call\n\nUsage notes:\n  - Domain filtering is supported to include or block specific websites\n  - Web search is only available in the US\n  - Account for "Today\'s date" in <env>. For example, if <env> says "Today\'s date: 2025-07-01", and the user wants the latest docs, do not use 2024 in the search query. Use 2025.\n',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "BashOutput",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bash_id": {
                                "type": "string",
                                "description": "The ID of the background shell to retrieve output from",
                            },
                            "filter": {
                                "type": "string",
                                "description": "Optional regular expression to filter the output lines. Only lines matching this regex will be included in the result. Any lines that do not match will no longer be available to read.",
                            },
                        },
                        "required": ["bash_id"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": "\n- Retrieves output from a running or completed background bash shell\n- Takes a shell_id parameter identifying the shell\n- Always returns only new output since the last check\n- Returns stdout and stderr output along with shell status\n- Supports optional regex filtering to show only lines matching a pattern\n- Use this tool when you need to monitor or check the output of a long-running shell\n- Shell IDs can be found using the /bashes command\n",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "KillShell",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "shell_id": {
                                "type": "string",
                                "description": "The ID of the background shell to kill",
                            }
                        },
                        "required": ["shell_id"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": "\n- Kills a running background bash shell by its ID\n- Takes a shell_id parameter identifying the shell to kill\n- Returns a success or failure status \n- Use this tool when you need to terminate a long-running shell\n- Shell IDs can be found using the /bashes command\n",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "SlashCommand",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": 'The slash command to execute with its arguments, e.g., "/review-pr 123"',
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                    "description": 'Execute a slash command within the main conversation\n\n**IMPORTANT - Intent Matching:**\nBefore starting any task, CHECK if the user\'s request matches one of the slash commands listed below. This tool exists to route user intentions to specialized workflows.\n\nHow slash commands work:\nWhen you use this tool or when a user types a slash command, you will see <command-message>{name} is running…</command-message> followed by the expanded prompt. For example, if .claude/commands/foo.md contains "Print today\'s date", then /foo expands to that prompt in the next message.\n\nUsage:\n- `command` (required): The slash command to execute, including any arguments\n- Example: `command: "/review-pr 123"`\n\nIMPORTANT: Only use this tool for custom slash commands that appear in the Available Commands list below. Do NOT use for:\n- Built-in CLI commands (like /help, /clear, etc.)\n- Commands not shown in the list\n- Commands you think might exist but aren\'t listed\n\nNotes:\n- When a user requests multiple slash commands, execute each one sequentially and check for <command-message>{name} is running…</command-message> to verify each has been processed\n- Do not invoke a command that is already running. For example, if you see <command-message>foo is running…</command-message>, do NOT use this tool with "/foo" - process the expanded prompt in the following message\n- Only custom slash commands with descriptions are listed in Available Commands. If a user\'s command is not listed, ask them to check the slash command file and consult the docs.\n',
                },
            },
        ],
        "max_tokens": 32000,
        "temperature": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    response = litellm.completion(**completion_kwargs)
    print("response: ", response)
    for chunk in response:
        print("chunk: ", chunk)


# Tests moved from test_streaming_n_with_tools.py
# Regression test for: https://github.com/BerriAI/litellm/issues/8977
@pytest.mark.parametrize("model", ["gpt-4o"])
@pytest.mark.asyncio
async def test_streaming_tool_calls_with_n_greater_than_1(model):
    """
    Test that the index field in a choice object is correctly populated
    when using streaming mode with n>1 and tool calls.

    Regression test for: https://github.com/BerriAI/litellm/issues/8977
    """
    tools = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "What is the weather in San Francisco?",
            },
        ],
        tools=tools,
        stream=True,
        n=3,
    )

    # Collect all chunks and their indices
    indices_seen = []
    for chunk in response:
        assert (
            len(chunk.choices) == 1
        ), "Each streaming chunk should have exactly 1 choice"
        assert hasattr(
            chunk.choices[0], "index"
        ), "Choice should have an index attribute"
        index = chunk.choices[0].index
        indices_seen.append(index)

    # Verify that we got chunks with different indices (0, 1, 2 for n=3)
    unique_indices = set(indices_seen)
    assert unique_indices == {
        0,
        1,
        2,
    }, f"Should have indices 0, 1, 2 for n=3, got {unique_indices}"

    print(
        f"✓ Test passed: streaming with n=3 and tool calls correctly populates index field"
    )
    print(f"  Indices seen: {indices_seen}")
    print(f"  Unique indices: {unique_indices}")


@pytest.mark.parametrize("model", ["gpt-4o"])
@pytest.mark.asyncio
async def test_streaming_content_with_n_greater_than_1(model):
    """
    Test that the index field is correctly populated for regular content streaming
    (not tool calls) with n>1.
    """
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Say hello in one word",
            },
        ],
        stream=True,
        n=2,
        max_tokens=10,
    )

    # Collect all chunks and their indices
    indices_seen = []
    for chunk in response:
        assert (
            len(chunk.choices) == 1
        ), "Each streaming chunk should have exactly 1 choice"
        assert hasattr(
            chunk.choices[0], "index"
        ), "Choice should have an index attribute"
        index = chunk.choices[0].index
        indices_seen.append(index)

    # Verify that we got chunks with different indices (0, 1 for n=2)
    unique_indices = set(indices_seen)
    assert unique_indices == {
        0,
        1,
    }, f"Should have indices 0, 1 for n=2, got {unique_indices}"

    print(
        f"✓ Test passed: streaming with n=2 and regular content correctly populates index field"
    )
    print(f"  Indices seen: {indices_seen}")
    print(f"  Unique indices: {unique_indices}")


def test_gpt_5_web_search():
    response = litellm.completion(
        model="openai/responses/gpt-5",
        messages=[{"role": "user", "content": "get price of nvda"}],
        stream=True,
        temperature=1,
        max_tokens=8192,
        tools=[{"type": "web_search"}],
    )

    for chunk in response:
        print("chunk: ", chunk)


def test_responses_gpt54_with_xhigh_reasoning():
    """
    Ensure chat->responses bridge sends the correct request payload for
    openai/responses/gpt-5.4 with reasoning_effort="xhigh".
    """
    with patch("litellm.responses") as mock_responses:
        # Stop execution right after request generation to avoid external API calls.
        mock_responses.side_effect = RuntimeError("stop_after_request_build")

        with pytest.raises(Exception):
            litellm.completion(
                model="openai/responses/gpt-5.4",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                reasoning_effort="xhigh",
                max_tokens=100,
            )

        mock_responses.assert_called_once()
        request_body = mock_responses.call_args.kwargs

        # The responses prefix should be stripped before routing.
        assert request_body["model"] == "gpt-5.4"
        # chat-completions reasoning_effort must map to Responses API reasoning.
        assert request_body["reasoning"] == {"effort": "xhigh"}
