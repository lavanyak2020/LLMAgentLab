from langchain_openai import ChatOpenAI
import asyncio
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing_extensions import Annotated, TypedDict
from browser_use import Agent, Controller
from browser_use import Browser
from pydantic import BaseModel

class TravelRequest(TypedDict):
    source: Annotated[str, 'Unknown']
    destination: Annotated[str, 'Unknown']
    date: Annotated[str, (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")]
    coach: Annotated[str, '3AC']

class Train(BaseModel):
    number: str
    name: str
    departure: str
    arrival: str
    duration: str
    availability: str
    fare: int

class TrainAvailabilityResponse(BaseModel):
    source: str
    destination: str
    date: str
    coach: str
    trains: List[Train]

load_dotenv()
controller = Controller(output_model = TrainAvailabilityResponse)
llm = ChatOpenAI(model = 'gpt-4o', temperature = 0.0)

def extract_parameters(query: str) -> Dict[str, Any]:
    prompt = f"""Extract the following travel details from the query:
    - source
    - destination
    - date (Default to tomorrow if not specified)
    - class (Default to 3AC if not specified)
    Query: "{query}"
    """

    structured_llm = llm.with_structured_output(TravelRequest)
    response = structured_llm.invoke(prompt)

    return response

async def railway_agent(user_query: str) -> Dict[str, Any]:
    travel_params = extract_parameters(user_query)
    prompt = f"""
        You're a Railway assistance agent. Go to IRCTC website and find all the trains available
        from {travel_params['source']} to {travel_params['destination']}
        on {travel_params['date']} in {travel_params['coach']} 
    """

    browser = Browser()
    agent = Agent(
        task="Find train availability",
        llm=llm,
        browser=browser,
        override_system_message=prompt,
        controller=controller,
        initial_actions= [
            {'open_tab': {'url': 'https://www.irctc.co.in/nget/train-search'}}
        ]
    )

    result = await agent.run()
    await browser.close()

    return result.final_result()

async def main():
    print("👮🏻‍♂️Welcome to Railway Booking Assistant! 👮🏻‍♂️")
    print("I can help you find train availability between cities in India.")
    print("Ask me questions like: 'Find trains from Delhi to Mumbai tomorrow for 3AC class'")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nWhat would you like to know about train availability? (or 'exit' to quit): ")

        if user_query in ['exit', 'quit', 'bye']:
            break
        print("\nSearching for train availability...")

        response = await railway_agent(user_query)
        print("\nResults:")
        print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
