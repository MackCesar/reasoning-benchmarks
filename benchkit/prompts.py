SYSTEM = "You are a careful reasoning assistant. Think step by step and provide FINAL: <answer>."

def cot(question: str) -> str:
    return f"{question}\n\nThink step by step, then provide 'FINAL: <answer>'."

def sc_base(question: str) -> str:
    return cot(question)

def tot_root(question: str) -> str:
    return (f"Problem: {question}\nPropose 2-3 short next steps, each under 2 sentences.")

def tot_refine(question: str, partial: str) -> str:
    return (f"Problem: {question}\nPartial:\n{partial}\nPropose 2-3 next steps to extend or correct.")