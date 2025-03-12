from typing import List
from googlesearch import search as _search
from bs4 import BeautifulSoup
from charset_normalizer import detect
import asyncio
from requests_html import AsyncHTMLSession
import urllib3
urllib3.disable_warnings()

from llama_cpp import Llama

# Load the model onto GPU
llama3 = Llama(
	"./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
	verbose=False,
	n_gpu_layers=-1,
	n_ctx=16384,    # This argument is how many tokens the model can take. The longer the better, but it will consume more memory. 16384 is a proper value for a GPU with 16GB VRAM.
)

def generate_response(_model: Llama, _messages: str) -> str:
	'''
	This function will inference the model with given messages.
	'''
	_output = _model.create_chat_completion(
		_messages,
		stop=["<|eot_id|>", "<|end_of_text|>"],
		max_tokens=512,    # This argument is how many tokens the model can generate, you can change it and observe the differences.
		temperature=0,      # This argument is the randomness of the model. 0 means no randomness. You will get the same result with the same input every time. You can try to set it to different values.
		repeat_penalty=2.0,
	)["choices"][0]["message"]["content"]
	return _output

async def worker(s:AsyncHTMLSession, url:str):
	try:
		header_response = await asyncio.wait_for(s.head(url, verify=False), timeout=10)
		if 'text/html' not in header_response.headers.get('Content-Type', ''):
			return None
		r = await asyncio.wait_for(s.get(url, verify=False), timeout=10)
		return r.text
	except:
		return None

async def get_htmls(urls):
	session = AsyncHTMLSession()
	tasks = (worker(session, url) for url in urls)
	return await asyncio.gather(*tasks)

async def search(keyword: str, n_results: int=3) -> List[str]:
	'''
	This function will search the keyword and return the text content in the first n_results web pages.

	Warning: You may suffer from HTTP 429 errors if you search too many times in a period of time. This is unavoidable and you should take your own risk if you want to try search more results at once.
	The rate limit is not explicitly announced by Google, hence there's not much we can do except for changing the IP or wait until Google unban you (we don't know how long the penalty will last either).
	'''
	keyword = keyword[:100]
	# First, search the keyword and get the results. Also, get 2 times more results in case some of them are invalid.
	results = list(_search(keyword, n_results * 2, lang="zh", unique=True))
	# Then, get the HTML from the results. Also, the helper function will filter out the non-HTML urls.
	results = await get_htmls(results)
	# Filter out the None values.
	results = [x for x in results if x is not None]
	# Parse the HTML.
	results = [BeautifulSoup(x, 'html.parser') for x in results]
	# Get the text from the HTML and remove the spaces. Also, filter out the non-utf-8 encoding.
	results = [''.join(x.get_text().split()) for x in results if detect(x.encode()).get('encoding') == 'utf-8']
	# Return the first n results.
	return results[:n_results]

def text_cufoff(text: str, max_length: int = 24000) -> str:
	if len(text) > max_length:
		text = text[:max_length]
	return text

class LLMAgent():
	def __init__(self, role_description: str, task_description: str, llm:str="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"):
		self.role_description = role_description   # Role means who this agent should act like. e.g. the history expert, the manager......
		self.task_description = task_description    # Task description instructs what task should this agent solve.
		self.llm = llm  # LLM indicates which LLM backend this agent is using.
	def inference(self, message:str) -> str:
		if self.llm == 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF': # If using the default one.
			# TODO: Design the system prompt and user prompt here.
			# Format the messsages first.
			messages = [
				{"role": "system", "content": f"{self.role_description}"},  # Hint: you may want the agents to speak Traditional Chinese only.
				{"role": "user", "content": f"{self.task_description}\n{message}"}, # Hint: you may want the agents to clearly distinguish the task descriptions and the user messages. A proper seperation text rather than a simple line break is recommended.
			]
			return generate_response(llama3, messages)
		else:
			# TODO: If you want to use LLMs other than the given one, please implement the inference part on your own.
			return ""
		
# This agent may help you filter out the irrelevant parts in question descriptions.
question_extraction_agent = LLMAgent(
	role_description="你專門負責從提供的文本或上下文中提取問題，刪除多餘背景，維持核心的問題。使用中文時只會使用繁體中文來回問題。",
	task_description=
		"從用戶提供的文本中提取問題，有下列要求：\n" + 
		"- 刪除多餘背景以及內容。\n" + 
		"- 禁止對文字進行改寫。\n" + 
		"- 禁止對文字進行補全。\n" +
		"- 開頭用【問題】"
)

# This agent may help you extract the keywords in a question so that the search tool can find more accurate results.
keyword_extraction_agent = LLMAgent(
	role_description="你專門提取問題的關鍵字，這些關鍵字用來對搜尋引擎檢索，以便找到更準確的答案。使用中文時只會使用繁體中文。",
	task_description=
		"從用戶提供的問題提取關鍵字，有下列要求：\n" +
		"- 每個關鍵字用空格分隔。\n" +
		"- 所有的關鍵字結合後與原本問題語意相同。\n" +
		"- 關鍵字保留所有的人物，地點，機構，時間，算式，歌曲，特殊名詞。\n" +
		"- 禁止回答問題。"
)

summary_agent = LLMAgent(
	role_description="你是答案生成專家，用戶將文本輸入後，根據提供的問題以及文本生成答案，不要使用自己的資料庫。使用中文時只會使用繁體中文。",
	task_description=
		"請生成文本的摘要，有下列要求：\n" +
		"- 僅根據問題生成摘要。\n" +
		"- 僅輸摘要內容。\n" +
		"- 禁止輸出問題。\n" +
		"- 禁止使用自己的資料庫。"
)

# This agent is the core component that answers the question.
qa_agent = LLMAgent(
	role_description="你是個回答問題專家，可以參考用戶提供的資料，或是以自己的資料庫回答。使用中文時只會使用繁體中文。",
	task_description=
		"請回答用戶的問題，會先提供問題，每個參考資料都以【參考資料】為開頭 ，有下列要求：\n" +
		"- 回答的內容請盡量精簡。\n" +
		"- 僅輸出答案。",
)

async def pipeline(question: str) -> str:
	# TODO: Implement your pipeline.
	# Currently, it only feeds the question directly to the LLM.
	# You may want to get the final results through multiple inferences.
	# Just a quick reminder, make sure your input length is within the limit of the model context window (16384 tokens), you may want to truncate some excessive texts.
	extracted_question = question_extraction_agent.inference(question)
	results = await search(extracted_question.split("】")[1], 3)
	results = [text_cufoff(result, 8000) for result in results]
	extracted_results = [summary_agent.inference(extracted_question + "\n" + result) for result in results]
	results = "".join([f"【參考資料】{extracted_result}\n" for extracted_result in extracted_results])
	final_answer = qa_agent.inference(f"{extracted_question}\n{results}")
	return final_answer

async def process():
	# Fill in your student ID first.
	STUDENT_ID = "r12921068"

	STUDENT_ID = STUDENT_ID.lower()
	with open('./public.txt', 'r') as input_f:
		questions = input_f.readlines()
		questions = [l.strip().split(',')[0] for l in questions]
		for id, question in enumerate(questions, 1):
			# if Path(f"./{STUDENT_ID}_{id}.txt").exists():
			#     continue
			answer = await pipeline(question)
			answer = answer.replace('\n',' ')
			print(id, answer)
			with open(f'./{STUDENT_ID}_{id}.txt', 'w') as output_f:
				print(answer, file=output_f)

	with open('./private.txt', 'r') as input_f:
		questions = input_f.readlines()
		for id, question in enumerate(questions, 31):
			# if Path(f"./{STUDENT_ID}_{id}.txt").exists():
			#     continue
			answer = await pipeline(question)
			answer = answer.replace('\n',' ')
			print(id, answer)
			with open(f'./{STUDENT_ID}_{id}.txt', 'w') as output_f:
				print(answer, file=output_f)

	with open(f'./{STUDENT_ID}.txt', 'w') as output_f:
		for id in range(1,91):
			with open(f'./{STUDENT_ID}_{id}.txt', 'r') as input_f:
				answer = input_f.readline().strip()
				print(answer, file=output_f)

if __name__ == "__main__":
	asyncio.run(process())