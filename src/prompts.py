from __future__ import annotations

REFUSAL_INSTRUCTION = (
    "If the Context is insufficient to support a conclusion, clearly output "
    "INSUFFICIENT_CONTEXT: <reason>."
)

LLM_SYSTEM = """You are a Building Technology assistant.

Answer the user's question directly using your own knowledge.
Do not use any external context or retrieval.
If you are uncertain, or the question requires evidence you do not have,
clearly output: INSUFFICIENT_CONTEXT: <reason>

Output format (keep the section titles exactly as written below):

1) Conclusion
2) Explanation
3) Confidence / Limits

Language rule:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
"""

LLM_STRUCTURED_SYSTEM = """You are a Building Technology assistant.

Answer the user's question directly using your own knowledge.
Do not use any external context or retrieval.
If the Context is insufficient to support a conclusion, clearly output
INSUFFICIENT_CONTEXT: <reason>.

Output format (keep the section titles exactly as written below):

1) Conclusion (1-2 sentences)
2) Explanation (bullet points)
3) Evidence
4) Assumptions & Scope
5) Suggested Next Retrieval

Language rule:
- Always answer in the same language as the user's question.
- Do NOT switch languages.
"""

JUDGE_SYSTEM = """You are an impartial evaluator for a Building Technology QA benchmark.

You will evaluate a model answer based on:
1. completeness
2. answer_relevance

Definitions:
- completeness: whether the answer covers the important information from the
  reference answer and key points. 1.0 = all covered, 0.5 = partially, 0.0 = missing.
- answer_relevance: whether the answer directly addresses the question.
  1.0 = fully relevant, 0.5 = partly, 0.0 = irrelevant.

Rules:
- Be strict but fair.
- Do not reward unsupported extra information.
- Ignore minor wording differences.
- Return JSON only.
"""

JUDGE_USER_TEMPLATE = """Question:
{question}

Reference answer:
{gold_answer}

Required key points:
{key_points}

Model answer:
{answer}

Return JSON only in this format:
{{
  "completeness": 0.0,
  "answer_relevance": 0.0,
  "reason": "short explanation"
}}
"""

FAITHFULNESS_SYSTEM = """You are an impartial factuality verifier.

Split the model answer into atomic claims. For each claim, decide whether the
provided Context supports it. Do not use outside knowledge.

Return JSON only:
{
  "claims": [{"claim": "...", "supported": true, "reason": "..."}],
  "supported_claims": 0,
  "total_claims": 0
}
"""

CITATION_SYSTEM = """You are an impartial citation checker.

You receive an answer, a list of citations ([C1] with source and snippet), and
the labels the answer cites. For each cited label, decide whether the cited
snippet actually supports the claim it is attached to.

Return JSON only:
{
  "cited_labels": ["C1"],
  "supported_labels": ["C1"],
  "unsupported": [{"label": "C2", "reason": "..."}]
}
"""

# Examples are drawn from the seven added HRI papers only. They must never be
# reused as benchmark questions (tests/test_fewshot_leakage.py enforces this).
COT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "What did the operating room study find about contaminant levels under laminar air flow in high-contamination areas?",
        "context": """[C1] Source: Experimental analysis of airborne contaminant distribution in an operating.pdf | page=1
With laminar air flow, higher contaminant levels are obtained in the areas with a high demand for contamination reduction.""",
        "answer": """1) Conclusion
Laminar air flow did not reduce contaminant levels in the most sensitive areas; it produced higher levels there than turbulent mixing ventilation.

2) Explanation
- The study compared laminar air flow and turbulent mixing ventilation in a research operating room.
- In areas with high hygiene requirements, laminar air flow led to higher contaminant levels.
- Laminar air flow was more effective for surgical smoke near the staff's facial area.

3) Evidence
- [C1]: With laminar air flow, higher contaminant levels were obtained in areas with a high demand for contamination reduction.

4) Assumptions & Scope
- This result is tied to the experimental setup and tracer aerosols used in the study.

5) Suggested Next Retrieval
operating room laminar air flow turbulent mixing ventilation contaminant concentration""",
    },
    {
        "question": "What is the main energetic finding of the UVC versus HEPA study?",
        "context": """[C1] Source: Energetic comparison of the inactivation of airborne corona viruses using.pdf | page=1
At the reference operating point, the energy demand for UVC-based inactivation exceeds that of filtration by up to a factor of 94.""",
        "answer": """1) Conclusion
Under the investigated conditions, UVC-based inactivation consumed far more electrical energy than HEPA filtration to achieve comparable virus reduction.

2) Explanation
- The study compares the electrical energy demand of UVC irradiation and HEPA filtration.
- At the reference point, UVC demand exceeds filtration by up to a factor of 94.
- The gap depends strongly on UVC source efficiency and chamber geometry.

3) Evidence
- [C1]: UVC-based inactivation exceeds filtration energy demand by up to a factor of 94.

4) Assumptions & Scope
- The result applies to the low-efficiency LEDs and chamber design used in the study.

5) Suggested Next Retrieval
UVC LED HEPA energy demand air purification virus inactivation""",
    },
    {
        "question": "What happens to paid heat cost when the nominal supply temperature is lowered from 40 to 35 C in HeatShift-HP?",
        "context": """[C1] Source: HeatShift-HP A Techno-Economic framework for assessing heat pump load.pdf | page=1
Lowering the nominal supply temperature from 40 C to 35 C reduces the median paid specific heat cost by about 3.7 % in Berlin and 4.4 % in Chicago for the standard controller.""",
        "answer": """1) Conclusion
Lowering the nominal supply temperature reduced the median paid specific heat cost in both cities.

2) Explanation
- The reduction was about 3.7% in Berlin and 4.4% in Chicago for the standard controller.
- Nominal system temperature was the dominant driver of both seasonal performance and paid heat cost.
- Additional storage mainly improved operational flexibility rather than lowering cost on its own.

3) Evidence
- [C1]: Lowering supply temperature from 40 C to 35 C reduced median paid heat cost by 3.7% in Berlin and 4.4% in Chicago.

4) Assumptions & Scope
- Results refer to the single-family reference building and the 2025 tariff assumptions.

5) Suggested Next Retrieval
heat pump supply temperature paid heat cost Berlin Chicago storage tariff""",
    },
    {
        "question": "What happens to the model ranking when only steady-state gas turbine operation is analyzed?",
        "context": """[C1] Source: Operational regime dependence of model performance and feature.pdf | page=1
The ranking reverses when steady-state operation is isolated, with Linear Regression achieving the lowest error of all four models.""",
        "answer": """1) Conclusion
The ranking reverses: once steady-state operation is isolated, Linear Regression performs best, although nonlinear models win on the mixed-regime dataset.

2) Explanation
- On the original mixed dataset, ANN and Random Forest had the lowest RMSE.
- After filtering to steady-state operation, Linear Regression achieved the lowest error.
- This regime-dependent reversal had not been reported previously.

3) Evidence
- [C1]: The ranking reverses under steady-state operation, with Linear Regression achieving the lowest error.

4) Assumptions & Scope
- The finding is based on roughly two years of data from one industrial gas turbine.

5) Suggested Next Retrieval
gas turbine steady-state operation model ranking linear regression RMSE""",
    },
    {
        "question": "What does Data2BEM automate and how much modeling time does it save?",
        "context": """[C1] Source: Automated building energy modeling for energy.pdf | page=2
The workflow integrates information extraction, model generation, and data-driven calibration.

[C2] Source: Automated building energy modeling for energy.pdf | page=2
Relative to professional practice, the system reduced total modeling time by over 90% (48 min versus 8-32 h) with minimal human input.""",
        "answer": """1) Conclusion
Data2BEM automates end-to-end building energy modeling and cuts modeling time by more than 90%.

2) Explanation
- It integrates information extraction, model generation, and data-driven calibration.
- On a real Cambridge office building it produced a calibrated model.
- Modeling time fell from 8-32 hours to about 48 minutes with minimal human input.

3) Evidence
- [C1]: The workflow integrates extraction, model generation, and calibration.
- [C2]: Total modeling time was reduced by over 90% (48 min versus 8-32 h).

4) Assumptions & Scope
- The time saving is measured against professional practice on one case-study building.

5) Suggested Next Retrieval
Data2BEM multi-agent building energy modeling calibration time saving""",
    },
]

NO_COT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "What did the operating room study find about contaminant levels under laminar air flow in high-contamination areas?",
        "context": COT_EXAMPLES[0]["context"],
        "answer": "Laminar air flow produced higher contaminant levels in areas with high hygiene requirements, although it reduced surgical smoke near staff faces.",
    },
    {
        "question": "What is the main energetic finding of the UVC versus HEPA study?",
        "context": COT_EXAMPLES[1]["context"],
        "answer": "UVC-based inactivation required up to 94 times more electrical energy than HEPA filtration at the reference operating point.",
    },
    {
        "question": "What happens to paid heat cost when the nominal supply temperature is lowered from 40 to 35 C in HeatShift-HP?",
        "context": COT_EXAMPLES[2]["context"],
        "answer": "The median paid specific heat cost fell by about 3.7% in Berlin and 4.4% in Chicago for the standard controller.",
    },
    {
        "question": "What happens to the model ranking when only steady-state gas turbine operation is analyzed?",
        "context": COT_EXAMPLES[3]["context"],
        "answer": "The ranking reverses: Linear Regression achieves the lowest error once steady-state operation is isolated.",
    },
    {
        "question": "What does Data2BEM automate and how much modeling time does it save?",
        "context": COT_EXAMPLES[4]["context"],
        "answer": "It automates information extraction, model generation, and calibration, cutting modeling time by over 90% (about 48 minutes versus 8-32 hours).",
    },
]

