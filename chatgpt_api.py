import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os
import glob
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')

# Add your own OpenAI API key
openai.api_key = CHATGPT_API_KEY

#def load_text(file_path):
#    with open(file_path, 'r') as file:
#        return file.read()

def read_texts_from_folder(folder_path):
    texts = []
    
    files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                texts.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return "\n\n".join(texts)

def save_to_file(responses, output_file):
    with open(output_file, 'w') as file:
        for response in responses:
            file.write(response + '\n')


def call_openai_api(chunk):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "PASS IN ANY ARBITRARY SYSTEM VALUE TO GIVE THE AI AN IDENITY"},
            {"role": "user", "content": f" Tóm tắt các đoạn văn dưới đây thành 1 đoạn văn có thể nói trong 8-10 phút. Lưu ý dịch đoạn tiếng anh sang tiếng việt và không nhất thiết phải tóm tắt theo thứ tự bài báo, có thể thêm các sự kiện, sự thật lịch sử để đoạn văn hay hơn.\n {chunk}."},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
    return chunks   

def process_chunks(input_file, output_file):
    text = read_texts_from_folder(input_file)
    chunks = split_into_chunks(text)
    
    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))

    save_to_file(responses, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read text from files in a folder and save to an array.')
    parser.add_argument('-d', '--directory', required=True, type=str, help='Path to the folder containing text files')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to the output file') 
    
    args = parser.parse_args()

    input_folder = args.directory
    output_file = args.output
    process_chunks(input_folder, output_file)

