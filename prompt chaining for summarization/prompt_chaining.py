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

# High-fidelity mock responses for fallback simulation
MOCK_SUMMARY = """# Detailed Summary of Agentic AI
Agentic AI marks a fundamental transition in artificial intelligence from passive, prompt-based assistants to active, autonomous entities capable of goal-oriented behavior.

### 1. Core Definition
Unlike traditional Large Language Model (LLM) systems that merely respond to static inputs, **Agentic AI** represents a paradigm shift toward autonomy. These systems are characterized by their ability to:
* **Perceive** and interpret their surrounding digital or physical environments.
* **Formulate** complex, multi-step plans.
* **Reason** through actions logically.
* **Interact** dynamically with external systems (such as databases, web browsers, and APIs) to independently achieve designated goals.

### 2. The Agentic Workflow Loop
At the heart of any agentic system is a continuous execution loop consisting of four main phases: **Observation, Planning, Execution, and Reflection**. 
The operational workflow follows these steps:
1. **Decomposition:** The agent receives a complex objective and breaks it down into smaller, manageable sub-tasks.
2. **Tool Selection & Execution:** The agent identifies and deploys the appropriate tool for a given sub-task and executes it.
3. **Inspection & Reflection:** The agent analyzes the output of the execution to determine if it brings the system closer to the final goal.
4. **Self-Correction:** If an action fails or yields unexpected results, the agent possesses the capability to self-correct, modify its plan, and attempt alternative approaches."""

MOCK_INSIGHTS = """Based on the summary of Agentic AI, here are 5 key actionable insights and takeaways:

* **Transition from static prompts to closed-loop workflows:** When designing AI systems, move beyond simple prompt-response models. Instead, implement a continuous loop of **Observation, Planning, Execution, and Reflection** so the system can autonomously decompose tasks, evaluate its own progress, and self-correct when errors occur.
* **Deploy advanced reasoning frameworks:** To help AI solve complex, multi-step problems, integrate structured reasoning techniques such as *Chain of Thought* (sequential, step-by-step processing) or *Tree of Thoughts* (exploring and evaluating multiple potential paths to a solution).
* **Build a dual-layered memory architecture:** Ensure your agentic systems maintain context and historical knowledge by pairing short-term memory (for managing immediate, in-context conversational states) with long-term memory (using vector databases to store and retrieve historical data over time).
* **Equip agents with dynamic tools:** Expand the utility of AI from simple text generation to real-world action by securely connecting agents to external tools, such as web search engines, file systems, code execution environments, and external APIs.
* **Establish strict operational guardrails:** To prevent runaway costs and security breaches, implement safety protocols that monitor agent behavior. Specifically, set limits to block infinite execution loops, restrict unauthorized system access, and cap resource consumption (API calls and compute power)."""

MOCK_BRIEF = """To drive next-generation operational efficiency, business leaders must transition AI from passive, prompt-response models to autonomous Agentic AI. By implementing closed-loop workflows—incorporating observation, planning, execution, and reflection—organizations enable AI to decompose complex tasks and self-correct. Scaling this capability requires integrating advanced reasoning frameworks, deploying dual-layered memory for long-term context, and securely connecting agents to dynamic external tools for real-world execution. Crucially, safeguarding these systems demands robust operational guardrails to prevent infinite loops, secure system access, and control resource costs. Ultimately, deploying secure, agentic AI transforms technology from a simple assistant into a highly capable, autonomous driver of business value."""

def generate_step(prompt_text, step_num, use_mock=False, model="gemini-3.5-flash"):
    """Helper function to run a single step in the prompt chain with simulation fallback."""
    if use_mock:
        if step_num == 1:
            return MOCK_SUMMARY
        elif step_num == 2:
            return MOCK_INSIGHTS
        else:
            return MOCK_BRIEF
            
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt_text
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Warning] Gemini API Error on step {step_num}: {e}. Falling back to simulation mode.")
        return generate_step(prompt_text, step_num, use_mock=True, model=model)

def run_pipeline(text):
    print("=" * 60)
    print("Starting Prompt Chaining Pipeline for Summarization")
    print("=" * 60)
    
    # Validation check on input
    if len(text.strip()) < 50:
        print("[Error] Input text is too short to summarize.")
        return
        
    print("\n--- Input Text Length: {} characters ---".format(len(text)))
    
    # Determine if we should start in mock mode
    use_mock = False
    if api_key == "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg" and not os.environ.get("GEMINI_API_KEY"):
        use_mock = True
        print("[System Info] Gemini API Key not set. Running in local high-fidelity simulation mode.")
        
    # Step 1: Detailed Summary
    print("\n[Step 1/3] Generating Detailed Summary...")
    prompt_1 = f"""
Analyze the following text and write a comprehensive, well-structured summary. Focus on capturing the core definitions, workflow loop, components, and challenges.

Text:
{text}

Detailed Summary:
"""
    summary = generate_step(prompt_1, 1, use_mock=use_mock)
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
    insights = generate_step(prompt_2, 2, use_mock=use_mock)
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
    brief = generate_step(prompt_3, 3, use_mock=use_mock)
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
