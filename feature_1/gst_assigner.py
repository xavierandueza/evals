import asyncio
from litellm import acompletion
from pydantic import BaseModel
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class GSTType(str, Enum):
    OUTPUT = "GST on Income"
    INPUT = "GST on Expenses"
    EXEMPTEXPENSES = "GST Free Expenses"
    EXEMPTOUTPUT = "GST Free Income"
    BASEXCLUDED = "BAS Excluded"
    GSTONIMPORTS = "GST on Imports"


class GSTAssignmentResponse(BaseModel):
    reasoning: str
    gst_type: GSTType


async def assign_gst_to_transaction(
    system_message: str, user_message: str, model: str, temperature: float
) -> GSTAssignmentResponse:
    response = await acompletion(
        model="openrouter/" + model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        response_format=GSTAssignmentResponse,
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content


if __name__ == "__main__":
    system_message = "You must assign the transaction to one of the gsy types and give reasoning"
    user_message = (
        "The transaction has description 'Lollies from supermarket' and was for price $10"
    )
    model = "deepseek/deepseek-chat-v3.1"

    task = assign_gst_to_transaction(
        system_message=system_message, user_message=user_message, model=model, temperature=0.0
    )
    asyncio.run(task)
