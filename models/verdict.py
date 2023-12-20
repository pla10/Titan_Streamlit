from pydantic import BaseModel, Field, validator
    
# Define your desired data structure.
class Verdict(BaseModel):
    verdict: str = Field(description="the verdict can be only “used” or “unused”")
    average: str = Field(description="average of the time series data")
    maximum: str = Field(description="maximum of the time series data")
    variance: str = Field(description="variance of the time series data")
    mode: str = Field(description="mode value of the time series data")
    comment: str = Field(description="give me a brief description of your thought process to decide the verdict. Given the fact that we are dealing with AWS EC2 instances, analyse the cpu data and decide if the instance is being used or not. Use common sense when deciding the thresholds.")

    # You can add custom validation logic easily with Pydantic.
    @validator("verdict")
    def is_valid(cls, value):
        if value != "used" and value != "unused":
            raise ValueError("Badly formed verdict!")
        return value