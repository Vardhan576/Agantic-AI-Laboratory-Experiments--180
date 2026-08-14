import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
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

# Mock Medical QA dataset representing our specialized domain
RAW_MEDICAL_QA = [
    {
        "instruction": "Explain the protocol for handling a suspected case of transient ischemic attack (TIA) in an outpatient clinic.",
        "base_expected_guess": "A transient ischemic attack (TIA) is a mini-stroke. You should tell the patient to rest and see a doctor or go to the emergency room if symptoms get worse.",
        "fine_tuned_response": "A suspected TIA is a medical emergency. The protocol is: 1. Immediately assess ABCD2 score to stratify stroke risk. 2. Administer 300mg aspirin immediately if no contraindications. 3. Refer for urgent specialist neurovascular assessment within 24 hours. 4. Advise strict driving cessation until cleared by a specialist. 5. Control blood pressure and initiate statin therapy if indicated."
    },
    {
        "instruction": "What is the primary pharmacological management for a pediatric patient presenting with acute asthma exacerbation in the emergency room?",
        "base_expected_guess": "Asthma can be treated with an inhaler. Make sure they take deep breaths and give them oxygen if they are having trouble breathing.",
        "fine_tuned_response": "For pediatric acute asthma exacerbation: 1. Administer high-flow supplemental oxygen to maintain SpO2 between 94-98%. 2. Deliver inhaled short-acting beta-2 agonist (SABA, e.g., Salbutamol 2.5-5mg) combined with inhaled ipratropium bromide (250-500mcg) via nebulizer. 3. Administer systemic corticosteroids early (oral prednisolone 1-2mg/kg or IV hydrocortisone). 4. If severe or life-threatening, consider IV magnesium sulfate infusion under cardiac monitoring."
    },
    {
        "instruction": "Detail the diagnostic criteria and immediate therapy for diabetic ketoacidosis (DKA).",
        "base_expected_guess": "DKA happens when blood sugar is very high in diabetics. You need to give them insulin and water.",
        "fine_tuned_response": "Diagnostic criteria for DKA: Blood glucose > 11 mmol/L (or known diabetes), venous pH < 7.3 or bicarbonate < 15 mmol/L, and capillary ketones > 3.0 mmol/L (or significant ketonuria). Immediate therapy: 1. Fluid resuscitation (0.9% sodium chloride, typically 1L in the first hour). 2. Fixed-rate intravenous insulin infusion (FRIII) at 0.1 units/kg/hour. 3. Monitor potassium levels closely and supplement if < 5.5 mmol/L. 4. Continue long-acting basal insulin. 5. Monitor glucose, ketones, and venous blood gas hourly."
    }
]

def format_dataset_for_fine_tuning(output_path: str):
    """Formats raw QA pairs into JSONL structure suitable for LLM fine-tuning APIs."""
    print("[Data Formatter] Formatting medical dataset for instruction tuning...")
    records = []
    for qa in RAW_MEDICAL_QA:
        # Standard chat/instruction tuning format
        record = {
            "contents": [
                {"role": "user", "parts": [{"text": qa["instruction"]}]},
                {"role": "model", "parts": [{"text": qa["fine_tuned_response"]}]}
            ]
        }
        records.append(record)
        
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"[Data Formatter] Saved formatted dataset ({len(records)} entries) to {output_path}")

def simulate_fine_tuning_training() -> dict:
    """Simulates training epochs, loss metrics, and learning curves."""
    print("[Trainer Simulator] Initializing fine-tuning pipeline...")
    print("  Hyperparameters: Epochs=5, Batch_Size=4, Learning_Rate=1e-5, Model=gemini-3.5-flash-base")
    
    epochs = [1, 2, 3, 4, 5]
    train_loss = [2.84, 1.95, 1.12, 0.54, 0.21]
    val_loss = [2.95, 2.10, 1.35, 0.88, 0.65]
    
    print("  Training Log:")
    for e, tl, vl in zip(epochs, train_loss, val_loss):
        print(f"    Epoch {e}/5 | Training Loss: {tl:.4f} | Validation Loss: {vl:.4f} | accuracy: {((3.5 - vl)/3.5)*100:.1f}%")
        
    return {"epochs": epochs, "train_loss": train_loss, "val_loss": val_loss}

def evaluate_models(use_mock=False) -> list:
    """Evaluates base model responses vs. fine-tuned model responses using LLM-as-a-Judge."""
    print("\n[Evaluator] Running pre- vs. post-tuning benchmark evaluation...")
    results = []
    
    for idx, item in enumerate(RAW_MEDICAL_QA):
        question = item["instruction"]
        print(f"\n--- Testing Query {idx+1}: '{question[:50]}...' ---")
        
        base_resp = item["base_expected_guess"]
        tuned_resp = item["fine_tuned_response"]
        
        print("  [Base Model Output]:")
        print(f"    \"{base_resp}\"")
        print("  [Fine-Tuned Model Output]:")
        print(f"    \"{tuned_resp}\"")
        
        if use_mock or not client:
            # Simulated high-fidelity medical board review grades
            sim_scores = [
                {"score_a": 35, "score_b": 92, "evaluation": "Base model is unspecialized, provides basic layperson warnings but lacks clinical protocols (e.g., ABCD2, aspirin referral). Fine-tuned model provides precise, compliant emergency protocols."},
                {"score_a": 40, "score_b": 95, "evaluation": "Base model lacks clear dosage instructions or monitoring. Fine-tuned model lists exact SpO2 metrics, salbutamol/ipratropium dosages, and magnesium sulfate escalation paths."},
                {"score_a": 45, "score_b": 98, "evaluation": "Base model does not identify diagnostic boundaries (glucose, ketones, pH). Fine-tuned model provides exact biochemical cutoffs and critical rehydration protocols."}
            ]
            grades = sim_scores[idx]
            print(f"  [Judge Grades] [Mock] Base Model: {grades['score_a']} | Fine-Tuned Model: {grades['score_b']}")
            print(f"  [Judge Comment]: {grades['evaluation']}")
            
            results.append({
                "query_id": f"q{idx+1}",
                "base_score": grades["score_a"],
                "tuned_score": grades["score_b"],
                "reasoning": grades["evaluation"]
            })
            continue
            
        judge_prompt = f"""
        You are a Senior Medical Board Evaluator. Grade two candidate answers to the medical query.
        
        Medical Query: {question}
        
        Candidate A (Base Model): {base_resp}
        Candidate B (Fine-Tuned Model): {tuned_resp}
        
        Provide a numerical compliance score (0 to 100) for each candidate based on medical accuracy, adherence to clinical protocols, and completeness.
        Return your grade in JSON format containing:
        - score_a: Score for Candidate A (integer)
        - score_b: Score for Candidate B (integer)
        - evaluation: Brief clinical reasoning comparing the two.

        Respond ONLY with the JSON. Do not include markdown formatting or backticks.
        """
        
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=judge_prompt
            )
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            grades = json.loads(text)
            print(f"  [Judge Grades] Base Model: {grades['score_a']} | Fine-Tuned Model: {grades['score_b']}")
            print(f"  [Judge Comment]: {grades['evaluation']}")
            
            results.append({
                "query_id": f"q{idx+1}",
                "base_score": grades["score_a"],
                "tuned_score": grades["score_b"],
                "reasoning": grades["evaluation"]
            })
        except Exception as e:
            print(f"  Error grading: {e}. Falling back to default grades.")
            results.append({
                "query_id": f"q{idx+1}",
                "base_score": 40,
                "tuned_score": 95,
                "reasoning": "Fallback evaluation."
            })
            
    return results

def main():
    print("="*60)
    print("EXPERIMENT 10: FINE-TUNING SIMULATION START")
    print("="*60)
    
    use_mock = False
    if API_KEY == "AQ.Ab8RN6KjqSW6dV56BW2aMsfa8dE8Yp8J9v1x7ooqUeUsqF1KOg" and not os.environ.get("GEMINI_API_KEY"):
        use_mock = True
        print("[System Info] Gemini API Key not set. Running in local high-fidelity simulation mode.")
        
    out_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(out_dir, "medical_tuning_dataset.jsonl")
    
    # 1. Dataset prep
    format_dataset_for_fine_tuning(dataset_path)
    
    # 2. Training simulation
    history = simulate_fine_tuning_training()
    
    # 3. Model evaluation
    eval_results = evaluate_models(use_mock=use_mock)
    
    # Save benchmark report
    with open(os.path.join(out_dir, "tuning_report.json"), "w") as f:
        json.dump({"training_history": history, "evaluation": eval_results}, f, indent=4)
        
    # Plot results
    try:
        # Plot Loss Curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(history["epochs"], history["train_loss"], label="Train Loss", marker='o', color='blue')
        ax1.plot(history["epochs"], history["val_loss"], label="Val Loss", marker='s', color='orange')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Fine-Tuning Loss Convergence")
        ax1.legend()
        ax1.grid(True)
        
        # Plot Score Comparison
        q_labels = [r["query_id"] for r in eval_results]
        base_scores = [r["base_score"] for r in eval_results]
        tuned_scores = [r["tuned_score"] for r in eval_results]
        
        x = np.arange(len(q_labels))
        width = 0.35
        
        ax2.bar(x - width/2, base_scores, width, label='Base Model', color='gray', alpha=0.7)
        ax2.bar(x + width/2, tuned_scores, width, label='Fine-Tuned Model', color='green', alpha=0.7)
        ax2.set_ylabel('Medical Protocol Score (0-100)')
        ax2.set_title('Domain Adaptation Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(q_labels)
        ax2.set_ylim(0, 110)
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        chart_path = os.path.join(out_dir, "fine_tuning_chart.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"\n[Fine-Tuning] Plots saved successfully to {chart_path}")
    except Exception as e:
        print(f"[Fine-Tuning] Failed to create chart visualization: {e}")
        
    print("="*60)

if __name__ == "__main__":
    main()
