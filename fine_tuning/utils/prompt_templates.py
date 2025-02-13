import numpy as np

SYSTEM_PROMPT_TEMPLATE = """
You are participating in a negotiation game.
There are {n_objects} objects: {object_names}.
Your valuation vector is: {v_agent}.
Rules:
- There will be {m_rounds} rounds of messaging.
- After messaging, both players simultaneously claim objects.
- For each object:
    • If only one player claims it, that player receives the full value.
    • If both claim it, each receives {p_fraction} of the value.
    • If neither claims it, each receives {q_fraction} of the value.
Your goal is to maximize your final payoff (the sum of the values you obtain).
Game Context: {game_context}.
Please negotiate with your opponent accordingly.
"""

def create_system_prompt(n_objects, v_agent, m_rounds, game_context,
                         p_fraction="1/4", q_fraction="1/2", object_names=None):
    if object_names is None:
        object_names = ", ".join([f"object_{i+1}" for i in range(n_objects)])
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        n_objects=n_objects,
        object_names=object_names,
        v_agent=np.array2string(v_agent, precision=2, separator=','),
        m_rounds=m_rounds,
        game_context=game_context,
        p_fraction=p_fraction,
        q_fraction=q_fraction
    )
    return prompt

if __name__ == '__main__':
    import numpy as np
    v_agent = np.array([1.23, -4.56, 7.89, 0.12])
    prompt = create_system_prompt(4, v_agent, 5, "Competitive Negotiation")
    print(prompt)
