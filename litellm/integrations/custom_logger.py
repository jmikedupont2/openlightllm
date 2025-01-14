#### What this does ####
#    On success, logs events to Promptlayer
import dotenv

from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache
from typing import Literal, Union, Optional
import traceback


class CustomLogger:  # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self) -> None:
        pass

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    #### ASYNC ####

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_pre_api_call(self, model, messages, kwargs):
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    #### PRE-CALL CHECKS - router/proxy only ####
    """
    Allows usage-based-routing-v2 to run pre-call rpm checks within the picked deployment's semaphore (concurrency-safe tpm/rpm checks).
    """

    async def async_pre_call_check(self, deployment: dict) -> Optional[dict]:
        pass

    def pre_call_check(self, deployment: dict) -> Optional[dict]:
        pass

    #### CALL HOOKS - proxy only ####
    """
    Control the modify incoming / outgoung data before calling the model
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Optional[
        Union[Exception, str, dict]
    ]:  # raise exception if invalid, return a str for the user to receive - if rejected, or return a modified dictionary for passing into litellm
        pass

    async def async_post_call_failure_hook(
        self, original_exception: Exception, user_api_key_dict: UserAPIKeyAuth
    ):
        pass

    async def async_post_call_success_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        pass

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation"],
    ):
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        pass

    #### SINGLE-USE #### - https://docs.litellm.ai/docs/observability/custom_callback#using-your-custom-callback-function

    def log_input_event(self, model, messages, kwargs, print_verbose, callback_func):
        try:
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["log_event_type"] = "pre_api_call"
            callback_func(
                kwargs,
            )
            print_verbose(f"Custom Logger - model call details: {kwargs}")
        except:
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")

    async def async_log_input_event(
        self, model, messages, kwargs, print_verbose, callback_func
    ):
        try:
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["log_event_type"] = "pre_api_call"
            await callback_func(
                kwargs,
            )
            print_verbose(f"Custom Logger - model call details: {kwargs}")
        except:
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")

    def log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose, callback_func
    ):
        # Method definition
        try:
            kwargs["log_event_type"] = "post_api_call"
            callback_func(
                kwargs,  # kwargs to func
                response_obj,
                start_time,
                end_time,
            )
        except:
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")

    async def async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose, callback_func
    ):
        # Method definition
        try:
            kwargs["log_event_type"] = "post_api_call"
            await callback_func(
                kwargs,  # kwargs to func
                response_obj,
                start_time,
                end_time,
            )
        except:
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")
