import time

from typing import Any
from typing import List
from pydantic import Field
from pydantic import BaseModel
from cflearn.api.cv.third_party.prompt import PromptEnhanceConfig

from .utils import api_pool
from .common import register_prompt_enhance
from .common import APIs
from .common import IAlgorithm
from .common import TextModel


txt2txt_prompt_enhance_endpoint = "/txt2txt/prompt_enhance"


class PromptEnhanceModel(TextModel):
    temperature: float = Field(
        0.9,
        description="a higher temperature will produce more diverse results, but with a higher risk of less coherent text.",
    )
    top_k: int = Field(
        8,
        description="the number of tokens to sample from at each step.",
    )
    max_length: int = Field(
        76,
        description="the maximum number of tokens for the output of the model.",
    )
    repitition_penalty: float = Field(
        1.2,
        description="the penalty value for each repetition of a token.",
    )
    num_return_sequences: int = Field(
        1,
        description="the number of results to generate.",
    )
    comma_mode: bool = Field(False, description="whether include more comma.")


class PromptEnhanceResponse(BaseModel):
    prompts: List[str] = Field(..., description="enhanced prompts.")


@IAlgorithm.auto_register()
class Txt2TxtPromptEnhance(IAlgorithm):
    model_class = TextModel
    response_model_class = PromptEnhanceResponse

    endpoint = txt2txt_prompt_enhance_endpoint

    def initialize(self) -> None:
        register_prompt_enhance()

    async def run(self, data: PromptEnhanceModel, *args: Any) -> PromptEnhanceResponse:
        self.log_endpoint(data)
        t0 = time.time()
        m = api_pool.get(APIs.PROMPT_ENHANCE)
        t1 = time.time()
        kw = data.dict()
        text = kw.pop("text")
        prompts = m.enhance(text, config=PromptEnhanceConfig(**kw))
        t2 = time.time()
        api_pool.cleanup(APIs.PROMPT_ENHANCE)
        self.log_times(
            {
                "get_model": t1 - t0,
                "inference": t2 - t1,
                "cleanup": time.time() - t2,
            }
        )
        return PromptEnhanceResponse(prompts=prompts)


__all__ = [
    "txt2txt_prompt_enhance_endpoint",
    "Txt2TxtPromptEnhance",
]
