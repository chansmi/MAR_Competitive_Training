import os
from utils.valuation_generation import generate_valuations
from utils.prompt_templates import create_system_prompt
from generate_data import parse_claims, compute_payoffs, generate_negotiation_transcript

def evaluate_models(n_evaluations=10):
    # For a held-out zero-sum test, for example:
    v1_test, v2_test = generate_valuations(4, theta=0.0)
    system_prompt = create_system_prompt(
        n_objects=4,
        v_agent=v1_test,
        m_rounds=5,
        game_context="Held-out Zero-Sum Test"
    )
    results = []
    for i in range(n_evaluations):
        transcript, claims = generate_negotiation_transcript(system_prompt, (v1_test, v2_test))
        claims_agent1 = parse_claims(claims["agent1"])
        claims_agent2 = parse_claims(claims["agent2"])
        payoff1, payoff2 = compute_payoffs(v1_test, v2_test, claims_agent1, claims_agent2)
        results.append((payoff1, payoff2))
    avg_payoffs = (sum([r[0] for r in results]) / n_evaluations,
                   sum([r[1] for r in results]) / n_evaluations)
    social_welfare = sum(avg_payoffs)
    print(f"Average Payoffs: {avg_payoffs}")
    print(f"Social Welfare: {social_welfare}")

if __name__ == '__main__':
    evaluate_models()
