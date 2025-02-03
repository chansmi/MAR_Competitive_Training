import os
import json

def prepare_agent_messages(input_file, output_file):
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            transcript = json.loads(line)
            # Extract only messages where role is 'agent'
            agent_messages = [msg for msg in transcript["messages"] if msg["role"].startswith("agent")]
            # Create a training example in the required format
            example = {"messages": agent_messages}
            json.dump(example, fout)
            fout.write("\n")

if __name__ == '__main__':
    datasets = ["competitive", "cooperative", "mixed"]
    for ds in datasets:
        input_file = os.path.join("data", "finetuning", f"{ds}.jsonl")
        output_file = os.path.join("data", "finetuning", f"{ds}_agent.jsonl")
        prepare_agent_messages(input_file, output_file)
        print(f"Prepared agent messages for {ds} dataset.")
