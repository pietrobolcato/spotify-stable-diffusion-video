# -*- coding: utf-8 -*-
"""Defines json schema for inference"""

from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """

    preset: str = Field(..., example="as_it_was", title="Preset name")
    init_image: str = Field(
        ..., example="https://www.server.com/image.jpg", title="Inital image to morph"
    )


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """

    elapsed_time: float = Field(
        ..., example=17.26, title="Required compute time for the generation"
    )
    video: str = Field(
        ..., example="base64", title="The generated video encoded in base64"
    )


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """

    error: bool = Field(..., example=False, title="Whether there is error")
    results: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """

    error: bool = Field(..., example=True, title="Whether there is error")
    message: str = Field(..., example="", title="Error message")
    traceback: str = Field(None, example="", title="Detailed traceback of the error")
