import time
import json
from jinja2 import Environment, FileSystemLoader
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    
)

deployment_name=os.getenv("GPT4O_MODEL_DEPLOYMENT_NAME")

class PromptEvaluator:
    """
    A class to evaluate and analyze prompts using a language model.
    Attributes:
    -----------
    env : Environment
        The Jinja2 environment for rendering templates.
    analytics_data : list
        A list to store performance data for each evaluation.
    Methods:
    --------
    __init__(template_dir):
        Initializes the PromptEvaluator with the given template directory.
    render_template(template_name, context={}):
        Renders a template with the given context.
    evaluate_prompt(evaluator_system_prompt, prompt_to_evaluate):
        Evaluates a prompt using the language model and logs performance data.
    log_performance(evaluator_system_prompt, prompt_to_evaluate, response, duration, clarity_score=None, relevance_score=None, completeness_score=None, role_score=None, context_score=None, examples_score=None, error=None):
        Logs the performance data of the prompt evaluation.
    evaluate_clarity(prompt_to_evaluate):
        Evaluates the clarity of a prompt and returns a numeric score.
    evaluate_relevance(prompt_to_evaluate):
        Evaluates the relevance of a prompt and returns a numeric score.
    evaluate_completeness(prompt_to_evaluate):
        Evaluates the completeness of a prompt and returns a numeric score.
    evaluate_role(prompt_to_evaluate):
        Evaluates the availability of role or identity in a prompt and returns a numeric score.
    evaluate_context(prompt_to_evaluate):
        Evaluates the availability of context or grounding data in a prompt and returns a numeric score.
    evaluate_examples(prompt_to_evaluate):
        Evaluates the availability of examples in a prompt and returns a numeric score.
    suggest_revised_prompt(prompt_to_evaluate):
        Suggests a revised version of a prompt based on prompt engineering best practices.
    generate_report():
        Generates a report summarizing the performance data of all evaluated prompts.
    log_error(error):
        Logs an error message.
    """
    def __init__(self, template_dir):
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.analytics_data = []

    def render_template(self, template_name, context={}):
        template = self.env.get_template(template_name)
        return template.render(context)

    def evaluate_prompt(self, evaluator_system_prompt, prompt_to_evaluate):
        start_time = time.time()
        try:

            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                {"role": "system", "content":evaluator_system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text":prompt_to_evaluate}
                    ],
                }
            ],
                temperature=0,
            )
            print("----Eval output start:---- ")
            print(response.choices[0].message.content)
            with open('evaluation_result.txt', 'w') as file:
                file.write(response.choices[0].message.content)            
            print("----Eval output end:---- ")

            end_time = time.time()
            clarity_score = self.evaluate_clarity(prompt_to_evaluate)
            relevance_score = self.evaluate_relevance(prompt_to_evaluate)
            completeness_score = self.evaluate_completeness(prompt_to_evaluate)
            role_score = self.evaluate_role(prompt_to_evaluate)
            context_score = self.evaluate_context(prompt_to_evaluate)
            examples_score = self.evaluate_examples(prompt_to_evaluate)
            #revised_prompt = self.suggest_revised_prompt(prompt_to_evaluate)
            self.log_performance(evaluator_system_prompt, prompt_to_evaluate, response, end_time - start_time, clarity_score, relevance_score, completeness_score, role_score, context_score, examples_score)
            return response.choices[0].message.content
        except Exception as e:
            end_time = time.time()
            self.log_performance(evaluator_system_prompt, prompt_to_evaluate, None, end_time - start_time, error=str(e))
            self.log_error(e)
            return None

    def log_performance(self, evaluator_system_prompt, prompt_to_evaluate, response, duration, clarity_score=None, relevance_score=None, completeness_score=None, role_score=None, context_score=None, examples_score=None,  error=None):
        performance_data = {
            "evaluator_system_prompt": evaluator_system_prompt,
            "prompt_to_evaluate": prompt_to_evaluate,
            "response": response,
            "duration": duration,
            "error": error,
            "clarity_score": clarity_score,
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "role_score": role_score,
            "context_score": context_score,
            "examples_score": examples_score,
            #"revised_prompt": revised_prompt
        }
        self.analytics_data.append(performance_data)

    def evaluate_clarity(self, prompt_to_evaluate):
        clarity_prompt = f"Rate the clarity of the following prompt on a scale of 1 to 10. Don't perform the task mentioned in prompt. Only rate it. Just return numeric answer, no text.:\n\n{prompt_to_evaluate}"
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
            {"role": "system", "content":clarity_prompt},
            {"role": "user", "content": [
                {"type": "text", "text":prompt_to_evaluate}
                ],
            }
        ],
            temperature=0,
        )
        print("Clarity response: " + response.choices[0].message.content)
        return float(response.choices[0].message.content.strip())

    def evaluate_relevance(self, prompt_to_evaluate):
        relevance_prompt = f"Rate the relevance of the following prompt on a scale of 1 to 10. Just return numeric answer, no text.:\n\nPrompt: {prompt_to_evaluate}"
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
            {"role": "system", "content":relevance_prompt},
            {"role": "user", "content": [
                {"type": "text", "text":prompt_to_evaluate}
                ],
            }
        ],
            temperature=0,
        )
        print("Relevance response " + response.choices[0].message.content)
        return float(response.choices[0].message.content.strip())

    def evaluate_completeness(self, prompt_to_evaluate):
        completeness_prompt = f"Based on the content of the response, determine the completeness of the response in addressing the prompt. If the response is incomplete or does not fully address the prompt, the rating should be low. If the response is complete and fully addresses the prompt, rate the completeness accordingly. Just do the rating. Just return the numeric answer, no text.:\n\nPrompt: {prompt_to_evaluate}"

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
            {"role": "system", "content":completeness_prompt},
            {"role": "user", "content": [
                {"type": "text", "text":prompt_to_evaluate}
                ],
            }
        ],
            temperature=0,
        )
        print("Completeness response: " + response.choices[0].message.content)
        return float(response.choices[0].message.content.strip())
    
    def evaluate_role(self, prompt_to_evaluate):
        role_prompt = f"Based on the content of the prompt, determine the availability of role or identity provided within it. If there is no role or identity, the rating should be 1. If there is role or identity, rate the availability accordingly. Just return the numeric answer, no text.:\n\nPrompt: {prompt_to_evaluate}"
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
            {"role": "system", "content":role_prompt},
            {"role": "user", "content": [
                {"type": "text", "text":prompt_to_evaluate}
                ],
            }
        ],
            temperature=0,
        )
        print(response.choices[0].message.content)
        return float(response.choices[0].message.content.strip())

    def evaluate_context(self, prompt_to_evaluate):
        context_prompt = f"Based on the content of the prompt, determine the availability of context or grounding data provided within it. If there is no context or grounding data, the rating should be 1. If there is context or grounding data, rate the availability accordingly. Just return the numeric answer, no text.:\n\nPrompt: {prompt_to_evaluate}"
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
            {"role": "system", "content":context_prompt},
            {"role": "user", "content": [
                {"type": "text", "text":prompt_to_evaluate}
                ],
            }
        ],
            temperature=0,
        )
        print(response.choices[0].message.content)
        return float(response.choices[0].message.content.strip())     

    def evaluate_examples(self, prompt_to_evaluate):
        examples_prompt = f"Based on the content of the prompt for GPT-4o, determine the availability of examples provided within it. If there are no examples, the rating should be 1. If there are examples, rate the availability accordingly. Just return the numeric answer, no text.:\n\nPrompt: {prompt_to_evaluate}"
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
            {"role": "system", "content":examples_prompt},
            {"role": "user", "content": [
                {"type": "text", "text":prompt_to_evaluate}
                ],
            }
        ],
            temperature=0,
        )
        print(response.choices[0].message.content)
        return float(response.choices[0].message.content.strip())     

    def suggest_revised_prompt(self, prompt_to_evaluate):
        revision_prompt = f"Suggest revised prompt for this prompt based on prompt engineering best practices. Do not perform the actual task asked in prompt. Just make the prompt itself better.:\n\nPrompt: {prompt_to_evaluate}"

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
            {"role": "system", "content":revision_prompt},
            {"role": "user", "content": [
                {"type": "text", "text":prompt_to_evaluate}
                ],
            }
        ],
            temperature=0,
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content

    def generate_report(self):
        # Example report generation
        report = {
            "total_requests": len(self.analytics_data),
            "successful_requests": len([d for d in self.analytics_data if d["error"] is None]),
            "failed_requests": len([d for d in self.analytics_data if d["error"] is not None]),
            "average_duration": sum(d["duration"] for d in self.analytics_data) / len(self.analytics_data) if self.analytics_data else 0,
            "average_clarity_score": sum(d["clarity_score"] for d in self.analytics_data) / len(self.analytics_data) if self.analytics_data else 0,
            "average_relevance_score": sum(d["relevance_score"] for d in self.analytics_data) / len(self.analytics_data) if self.analytics_data else 0,
            "average_completeness_score": sum(d["completeness_score"] for d in self.analytics_data) / len(self.analytics_data) if self.analytics_data else 0,
            "average_role_score": sum(d["role_score"] for d in self.analytics_data) / len(self.analytics_data) if self.analytics_data else 0,
            "average_context_score": sum(d["context_score"] for d in self.analytics_data) / len(self.analytics_data) if self.analytics_data else 0,
            "average_examples_score": sum(d["examples_score"] for d in self.analytics_data) / len(self.analytics_data) if self.analytics_data else 0,
            "errors": [d["error"] for d in self.analytics_data if d["error"] is not None],
            #"revised_prompt": [d["revised_prompt"] for d in self.analytics_data if d["revised_prompt"] is not None]
        }
        return json.dumps(report, indent=4)

    def log_error(self, error):
        # Implement logging logic here
        print(f"Error: {error}")

# Example usage
if __name__ == "__main__":
    evaluator = PromptEvaluator(template_dir='.')
    evaluator_system_prompt = evaluator.render_template('eval_system_prompt_template.j2')
    prompt_to_evaluate = evaluator.render_template('prompt_to_evaluate.j2')
    try:
        result = evaluator.evaluate_prompt(evaluator_system_prompt, prompt_to_evaluate)
        print(result)
    except Exception as e:
        evaluator.log_error(e)

    # Generate and print report
    report = evaluator.generate_report()
    print(report)