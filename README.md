# Open Your Eyes (OYE) 
## Vision LLMs Lab  
Lab to explore LLM vision capabilities  

# How to configure Prompt Evaluator?  
1. Create a .env file with following environment variables and their values:  
OPENAI_API_KEY=api key for Azure OpenAI (AOAI) service  
OPENAI_API_BASE=Endpoint for AOAI service  
OPENAI_API_VERSION=AOAI model API version  
GPT4O_MODEL_DEPLOYMENT_NAME=AOAI model deployment name  

2. Save eval_system_prompt_template.j2.sample file as eval_system_prompt_template.j2 . You can customize the evaluation prompt which acts as a system prompt and is used to evaluate the source prompt.  

3. Save prompt_to_evaluate.j2.sample as prompt_to_evaluate.j2 . This will contain the prompt you want to evaluate.  

# How to run Prompt Evaluator? 
Run it with "python prompt-evaluator.py"  

# What to expect as Prompt Evaluator output?  
There are 2 responses you can expect.  
1. A file named evaluation_result.txt will be created which contains prompt review qualitative details and the suggested prompt revision.  
2. On console, it will show a report with score, e.g., shown below:  

    "average_clarity_score": 9.0,  
    "average_relevance_score": 10.0,  
    "average_completeness_score": 5.0,  
    "average_role_score": 3.0,  
    "average_context_score": 3.0,   

