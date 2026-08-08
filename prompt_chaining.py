import os
from google import genai

# Setup Gemini Client
api_key = os.environ.get("GEMINI_API_KEY") or "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg"
client = genai.Client(api_key=api_key)

DEFAULT_TEXT = """
Agentic AI represents a paradigm shift in artificial intelligence, moving from passive assistants to active, autonomous agents. Unlike traditional LLM systems that simply respond to static prompts, Agentic AI systems are designed to perceive their environment, create multi-step plans, reason about actions, and interact with external tools (such as databases, web browsers, and APIs) to achieve specific goals.

At the core of an agentic workflow is the loop of observation, planning, execution, and reflection. When given a complex goal, the agent first decomposes the task into smaller sub-tasks. It then selects the appropriate tool for the job, runs it, inspects the output, and reflections on whether the result brings it closer to the goal. If a tool execution fails or produces unexpected results, the agent can self-correct, adjust its plan, and try an alternative approach.

Key components of agentic architectures include:
1. Planning: The ability to break down goals and perform self-reflection or critique (e.g., Chain of Thought, Tree of Thoughts).
2. Memory: Short-term memory (in-context learning and conversational state) and Long-term memory (vector databases for retrieving historical interactions).
3. Tools: Capabilities to execute code, search the web, read files, or call APIs to perform actions in the physical or digital world.

This technology has wide-ranging applications, from automated software engineering and automated data pipelines to personalized learning companions and autonomous scientific research. However, agentic workflows also raise important challenges, including safety, security, alignment, predictability, and the prevention of infinite execution loops or resource exhaustion.
"""

def generate_step(prompt_text, model="gemini-3.5-flash"):
    """Helper function to run a single step in the prompt chain."""
    response = client.models.generate_content(
        model=model,
        contents=prompt_text
    )
    return response.text.strip()

def run_pipeline(text):
    print("=" * 60)
    print("Starting Prompt Chaining Pipeline for Summarization")
    print("=" * 60)
    
    # Validation check on input
    if len(text.strip()) < 50:
        print("[Error] Input text is too short to summarize.")
        return
        
    print("\n--- Input Text Length: {} characters ---".format(len(text)))
    
    # Step 1: Detailed Summary
    print("\n[Step 1/3] Generating Detailed Summary...")
    prompt_1 = f"""
Analyze the following text and write a comprehensive, well-structured summary. Focus on capturing the core definitions, workflow loop, components, and challenges.

Text:
{text}

Detailed Summary:
"""
    summary = generate_step(prompt_1)
    print("\n===== STEP 1 OUTPUT: DETAILED SUMMARY =====")
    print(summary)
    print("===========================================")
    
    # Validation Check 1
    if not summary:
        print("[Error] Step 1 failed to produce an output.")
        return

    # Step 2: Extract Key Takeaways (using summary as input)
    print("\n[Step 2/3] Extracting Key Insights...")
    prompt_2 = f"""
Read the following summary and extract 5 key actionable insights or takeaways. Format them as a bulleted list.

Summary:
{summary}

Key Takeaways:
"""
    insights = generate_step(prompt_2)
    print("\n===== STEP 2 OUTPUT: KEY INSIGHTS =====")
    print(insights)
    print("=======================================")

    # Validation Check 2
    if not insights:
        print("[Error] Step 2 failed to produce an output.")
        return

    # Step 3: Executive Synthesis (using insights as input)
    print("\n[Step 3/3] Synthesizing into Executive Brief...")
    prompt_3 = f"""
Based on the key insights below, write a professional, high-level executive brief suitable for business leaders.
The brief must be a single, cohesive paragraph of no more than 120 words. Focus on the strategic impact.

Key Insights:
{insights}

Executive Brief:
"""
    brief = generate_step(prompt_3)
    print("\n===== STEP 3 OUTPUT: EXECUTIVE BRIEF =====")
    print(brief)
    print("==========================================")
    
    print("\n" + "=" * 25 + " Pipeline Execution Complete " + "=" * 25 + "\n")

def main():
    text_input = input("Enter text to summarize (or press enter to use default agentic AI article): ")
    if not text_input.strip():
        text_input = DEFAULT_TEXT
    run_pipeline(text_input)

if __name__ == "__main__":
    main()
