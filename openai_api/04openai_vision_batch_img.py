from openai import OpenAI
import json

# https://cookbook.openai.com/examples/batch_processing

OPENAI_API_KEY= 'your-api-key'

client = OpenAI(api_key=OPENAI_API_KEY)

batch_input_file = client.files.create(
  file=open("batch_files.jsonl", "rb"),
  purpose="batch"
)

batch_input_file_id = batch_input_file.id

batch_object_return = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    }
)



batch_job = client.batches.retrieve(batch_object_return.id)
print(batch_job.status)

# stop here until the batch job is completed
import ipdb; ipdb.set_trace()

result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content


result_file_name = "./batch_job_results_movies.jsonl"

with open(result_file_name, 'wb') as file:
    file.write(result)



results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)

# Reading only the first results
for res in results[:5]:
    task_id = res['custom_id']
    # Getting index from task id
    index = task_id.split('-')[-1]
    result = res['response']['body']['choices'][0]['message']['content']
    print(f"RESULT: {result}")
    print("\n\n----------------------------\n\n")