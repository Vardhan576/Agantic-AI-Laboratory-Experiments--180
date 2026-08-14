import os
from google import genai

# Setup Gemini Client
API_KEY = os.environ.get("GEMINI_API_KEY") or "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg"
client = None
if os.environ.get("GEMINI_API_KEY") or API_KEY != "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg":
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception:
        client = None

MODEL_NAME = "gemini-3.5-flash"

def planner_agent(topic: str, use_mock=False) -> str:
    """Agent 1: Generates an outline and sub-questions for research."""
    print(f"[Planner Agent] Creating research plan for topic: '{topic}'...")
    
    if use_mock or not client:
        print("[Planner Agent] [Mock] Generating outline...")
        return """# Research Outline: Agentic AI in Healthcare
## Section 1: Clinical Decision Support & Diagnostics
- Sub-point A: How multi-agent LLM systems aid doctors in diagnosing rare pathologies.
- Sub-point B: Safety protocols for validating LLM diagnoses against medical standards.
## Section 2: Administrative Automation & Patient Routing
- Sub-point A: Automating clinical charting and speech-to-text EHR updates.
- Sub-point B: Intelligent triage systems to optimize emergency room queues.
## Section 3: Safety, Compliance, and Ethical Guardrails
- Sub-point A: Enforcing HIPAA and GDPR compliance for agent-hosted medical data.
- Sub-point B: The role of human-in-the-loop validation to prevent hallucinated prescriptions.
**Target Audience**: Executive healthcare leaders and clinical operations directors.
**Tone**: Academic, objective, and risk-sensitive."""

    prompt = f"""
    You are a Research Planner Agent. Your job is to create a structured outline for a short research report about: '{topic}'.
    The outline must include:
    - 3 specific sections with titles
    - 2 core sub-points/questions to address in each section
    - Target tone and audience definition (e.g., professional, executive)

    Respond with clean markdown format. Do not use any introductory conversational filler.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Planner Agent] Warning: API Error ({e}). Using mock planner output.")
        return planner_agent(topic, use_mock=True)

def drafting_agent(topic: str, plan: str, use_mock=False) -> str:
    """Agent 2: Writes the initial draft based on the research plan."""
    print(f"[Drafting Agent] Generating initial draft based on plan...")
    
    if use_mock or not client:
        print("[Drafting Agent] [Mock] Generating initial report draft...")
        return """# The Role of Agentic AI in Modern Healthcare Systems

## Introduction
Healthcare systems globally are facing unprecedented challenges, including clinician burnout and rising operational costs. Agentic AI workflows, characterized by proactive planning, tool use, and multi-agent coordination, present a promising solution to these bottlenecks.

## Section 1: Clinical Decision Support & Diagnostics
Clinicians spend hours analyzing diagnostic data. Multi-agent LLM pipelines can cross-reference patient symptoms against medical databases in seconds. For instance, a diagnostic agent can consult a research agent to identify rare syndromes. However, validation remains a bottleneck. LLMs are prone to hallucinations, so safety checks must cross-reference guidelines like the DSM-5.

## Section 2: Administrative Automation & Patient Routing
Administrative duties account for nearly a third of all healthcare costs. LLM agents can transcribe doctor-patient conversations and automatically write EHR records. In routing, triage agents can analyze triage forms to prioritize patients. This optimizes emergency room efficiency.

## Section 3: Safety, Compliance, and Ethical Guardrails
Because clinical data is sensitive, HIPAA and GDPR compliance must be enforced. Any agent workflow must run in a secure, isolated sandbox to prevent leaks. Moreover, human-in-the-loop validation is vital. No agent should prescribe medication without direct MD approval to prevent catastrophic errors.

## Conclusion
Agentic AI can transform healthcare diagnostics and administration, provided that strict security boundaries and medical evaluations are integrated into the pipeline."""

    prompt = f"""
    You are a Research Drafting Agent. Write a detailed research report on the topic: '{topic}' following the plan/outline below.
    
    Research Outline:
    {plan}
    
    Constraints:
    - Write structured markdown.
    - Provide deep, descriptive content for each section.
    - Keep the length between 400 and 800 words.
    - Include a brief introduction and a conclusion.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Drafting Agent] Warning: API Error ({e}). Using mock draft.")
        return drafting_agent(topic, plan, use_mock=True)

def reflector_agent(draft: str, use_mock=False) -> str:
    """Agent 3: Reflects on the generated draft, providing critical feedback."""
    print(f"[Reflector Agent] Analyzing draft and compiling reflection feedback...")
    
    if use_mock or not client:
        print("[Reflector Agent] [Mock] Compiling critique feedback...")
        return """# Peer Review Feedback on Medical AI Draft
1. **Depth & Examples**: The diagnostic section states that agents 'cross-reference symptoms,' but lacks a specific clinical example (e.g., diagnosing stroke via automated scan alerts).
2. **Formatting**: Section 1 mentions the DSM-5, which is psychological. For diagnostics, citing guidelines like NICE or ACC/AHA is more appropriate.
3. **Data Security Details**: Section 3 mentions HIPAA, but does not detail *how* agents protect data (e.g., using federated learning or differential privacy).
4. **Overall Assessment**: The draft is coherent but lacks specific medical credibility. Add concrete examples and technical terms like 'de-identification' and 'clinical decision support systems'."""

    prompt = f"""
    You are a Peer Reviewer and Reflector Agent. Critique the research draft below.
    Identify strengths, weaknesses, and clear actionable recommendations for improvement.
    
    Focus on:
    - Does it cover all sections specified in the outline?
    - Are the technical details accurate and clear?
    - Is the tone professional and consistent?
    - Where does it lack evidence, depth, or formatting clarity?
    
    Research Draft:
    ---
    {draft}
    ---
    
    Provide your feedback in structured markdown format. Be specific and detailed in your recommendations.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Reflector Agent] Warning: API Error ({e}). Using mock feedback.")
        return reflector_agent(draft, use_mock=True)

def refinement_agent(original_draft: str, feedback: str, use_mock=False) -> str:
    """Agent 4: Revises the draft by incorporating reflection feedback."""
    print(f"[Refinement Agent] Revising and refining draft based on reflection...")
    
    if use_mock or not client:
        print("[Refinement Agent] [Mock] Applying revisions to report...")
        return """# The Role of Agentic AI in Modern Healthcare Systems (Revised Edition)

## Introduction
Healthcare systems globally are facing unprecedented challenges, including clinician burnout and rising operational costs. Agentic AI workflows, characterized by proactive planning, tool use, and multi-agent coordination, present a promising solution to these bottlenecks.

## Section 1: Clinical Decision Support & Diagnostics
Clinicians spend hours analyzing complex diagnostic scans and lab results. Multi-agent Clinical Decision Support Systems (CDSS) can cross-reference patient symptoms against medical databases in seconds. For instance, an automated imaging agent can flag suspected strokes in head CT scans, alerting neurologist agents immediately. To ensure clinical safety, these systems are validated against guidelines established by organizations like the ACC (American College of Cardiology) and AHA (American Heart Association), preventing diagnostic drift.

## Section 2: Administrative Automation & Patient Routing
Administrative duties account for nearly a third of all healthcare costs. LLM agents can transcribe doctor-patient conversations and automatically write EHR records. In routing, triage agents can analyze triage forms to prioritize patients. This optimizes emergency room efficiency.

## Section 3: Safety, Compliance, and Ethical Guardrails
Because clinical data is highly sensitive, strict HIPAA and GDPR compliance must be enforced. Agentic pipelines protect Patient Health Information (PHI) by applying automated de-identification pipelines and differential privacy techniques before querying external models. Any agent workflow must run in a secure, isolated sandbox to prevent leaks. Moreover, human-in-the-loop validation is vital. No agent should prescribe medication without direct MD approval to prevent catastrophic errors.

## Conclusion
Agentic AI can transform healthcare diagnostics and administration, provided that strict security boundaries and medical evaluations are integrated into the pipeline."""

    prompt = f"""
    You are a Master Editor and Refinement Agent. Revise the original research draft by thoroughly incorporating the reviewer's feedback.
    
    Original Draft:
    ---
    {original_draft}
    ---
    
    Reviewer Feedback:
    ---
    {feedback}
    ---
    
    Deliver the final, polished version of the research report in clean markdown format. Ensure all issues pointed out in the feedback are fixed.
    """
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Refinement Agent] Warning: API Error ({e}). Using mock refined report.")
        return refinement_agent(original_draft, feedback, use_mock=True)

def main():
    print("="*60)
    print("EXPERIMENT 7: DEEP RESEARCH AGENT WORKFLOW START")
    print("="*60)
    
    use_mock = False
    if API_KEY == "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg" and not os.environ.get("GEMINI_API_KEY"):
        use_mock = True
        print("[System Info] Gemini API Key not set. Running in local high-fidelity simulation mode.")
        
    topic = "The Role of Agentic AI in Modern Healthcare Systems"
    
    # Step 1: Planning
    plan = planner_agent(topic, use_mock=use_mock)
    print("\n--- GENERATED PLAN ---")
    print(plan)
    print("----------------------")
    
    # Step 2: Drafting
    draft = drafting_agent(topic, plan, use_mock=use_mock)
    print("\n--- INITIAL DRAFT ---")
    print(draft[:300] + "\n... [truncated] ...\n" + draft[-200:])
    print("----------------------")
    
    # Step 3: Reflection
    feedback = reflector_agent(draft, use_mock=use_mock)
    print("\n--- REVIEWER FEEDBACK ---")
    print(feedback)
    print("-------------------------")
    
    # Step 4: Refinement
    final_report = refinement_agent(draft, feedback, use_mock=use_mock)
    print("\n--- FINAL REFINED REPORT ---")
    print(final_report[:500] + "\n... [truncated] ...\n" + final_report[-300:])
    print("----------------------------")
    
    # Save the output documents
    out_dir = os.path.dirname(__file__)
    
    with open(os.path.join(out_dir, "research_plan.md"), "w", encoding="utf-8") as f:
        f.write(plan)
        
    with open(os.path.join(out_dir, "initial_draft.md"), "w", encoding="utf-8") as f:
        f.write(draft)
        
    with open(os.path.join(out_dir, "reflector_feedback.md"), "w", encoding="utf-8") as f:
        f.write(feedback)
        
    with open(os.path.join(out_dir, "final_research_report.md"), "w", encoding="utf-8") as f:
        f.write(final_report)
        
    print(f"\n[Research Workflow] All outputs (plan, draft, feedback, final report) saved to {out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
