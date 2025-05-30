import vertexai
from vertexai.language_models import ChatModel
import argparse

PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
MODELS = ["chat-bison", "gemini-pro", "gemini-1.5-flash-preview-0514"]

DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What are the pros and cons of remote work?",
    "Describe how photosynthesis works.",
    "Give me a beginner's explanation of machine learning.",
    "What's the importance of mental health in schools?"
]

def initialize_vertex():
    vertexai.init(project=PROJECT_ID, location=LOCATION)

def fetch_response(model_name, prompt):
    try:
        chat_model = ChatModel.from_pretrained(model_name)
        chat = chat_model.start_chat()
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error from {model_name}: {str(e)}"

def grade_single_prompt(prompt, responses):
    grading_prompt = f"You are grading different AI responses to the following prompt:\n\nPROMPT: {prompt}\n\n"
    for i, res in enumerate(responses):
        grading_prompt += f"RESPONSE {i+1} ({MODELS[i]}):\n{res}\n\n"
    grading_prompt += "Give a score out of 10 for each response with reasoning.\n"

    grader_model = ChatModel.from_pretrained("chat-bison")
    grader = grader_model.start_chat()
    result = grader.send_message(grading_prompt)
    return result.text

def run_evaluation(prompts):
    initialize_vertex()
    all_responses = {model: [] for model in MODELS}
    all_scores = {model: [] for model in MODELS}

    for prompt in prompts:
        print(f"\n🟦 Prompt: {prompt}\n" + "="*60)
        prompt_responses = []
        for model in MODELS:
            print(f"\n--- Querying {model} ---")
            response = fetch_response(model, prompt)
            print(f"📝 Response from {model}:\n{response}\n" + "-"*60)
            all_responses[model].append(response)
            prompt_responses.append(response)

        print("\n🧠 Grading responses...")
        grading_result = grade_single_prompt(prompt, prompt_responses)
        print(f"\n🏅 Grading Result:\n{grading_result}\n" + "="*60)

        for i, model in enumerate(MODELS):
            try:
                lines = grading_result.splitlines()
                for line in lines:
                    if model in line:
                        score = int(''.join(filter(str.isdigit, line)))
                        all_scores[model].append(score)
                        break
            except:
                all_scores[model].append(0)

    print("\n✅ === Final Aggregated Results ===\n")
    for model in MODELS:
        scores = all_scores[model]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"{model} - Avg Score: {avg_score:.2f} from {len(scores)} prompts.")
        print(f"Scores: {scores}\n")

def main():
    parser = argparse.ArgumentParser(description="Compare multiple GCP models across multiple prompts and grade them.")
    parser.add_argument("--use-default", action="store_true", help="Use default prompt list")
    parser.add_argument("--prompt-file", type=str, help="Path to file with one prompt per line")
    parser.add_argument("--prompt", type=str, help="Single prompt to evaluate")

    args = parser.parse_args()

    if args.use_default:
        prompts = DEFAULT_PROMPTS
    elif args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        print("❌ Please provide one of: --use-default, --prompt-file, or --prompt.")
        return

    run_evaluation(prompts)

if __name__ == "__main__":
    main()
