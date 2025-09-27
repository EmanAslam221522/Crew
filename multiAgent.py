"""
CrewAI Multi-Agent System with Gemini
=====================================

- Takes a topic from user input
- Research Analyst researches the topic
- Copywriter creates engaging copy from the research
- Uses Google Gemini API as the LLM
- API key loaded from .env file

Author: [Your Name]
Date: September 2025
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# =============================================================================
# Configuration
# =============================================================================

# Load environment variables from .env
load_dotenv()

# Read Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in .env file")

# Initialize Gemini LLM
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.2
)

# =============================================================================
# Agent Definitions
# =============================================================================

researcher = Agent(
    role="Research Analyst",
    goal="Analyze and research topics thoroughly.",
    backstory="An experienced analyst who finds key insights quickly and clearly.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

copywriter = Agent(
    role="Copywriter",
    goal="Write engaging and professional content.",
    backstory="A skilled writer who transforms research into compelling copy.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("🚀 CrewAI Multi-Agent System with Gemini")
    print("========================================")
    
    # Take input topic from user
    topic = input("📝 Enter a topic for research: ")

    # Define tasks dynamically based on input
    research_task = Task(
        description=f"Research the topic: {topic}. Provide a clear and concise summary including key insights, recent developments, and implications.",
        agent=researcher,
        expected_output=f"A detailed research summary on {topic}, maximum 120 words."
    )

    copywriting_task = Task(
        description=f"Based on the research about {topic}, write professional and engaging marketing copy that highlights opportunities and benefits.",
        agent=copywriter,
        expected_output=f"Marketing copy that explains the value of {topic} in a persuasive and professional way."
    )

    # Setup crew
    crew = Crew(
        agents=[researcher, copywriter],
        tasks=[research_task, copywriting_task],
        process=Process.sequential,
        verbose=True
    )

    # Run the workflow
    result = crew.kickoff()

    print("\n🎯 FINAL RESULT")
    print("="*50)
    print(result)
    print("="*50)
    print("✅ Workflow completed successfully!")

if __name__ == "__main__":
    main()
