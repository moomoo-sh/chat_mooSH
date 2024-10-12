# Original : https://github.com/VictorTaelin/ChatSH

import sys
import re
import subprocess
import platform
import argparse
from os import environ

import llama_cpp

class bcolors:
    CHATC = '\033[38;5;12m' #LIGHTBLUE
    SYSC = '\033[38;5;244m' #LIGHTGREY
    ENDC = '\033[0m'

parser = argparse.ArgumentParser(description="Run ChatSH with custom model and context window size")
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--n_ctx', type=int, default=6144, help='Context window size (default: 6144 tokens)')
args = parser.parse_args()

system_message = """You are ChatSH, an AI language model that specializes in assisting users with tasks on their system using shell commands, AND chatting or answering open-ended questions via the terminal.

# CHATSH GUIDE:

- If the USER asks you to perform a SYSTEM TASK, ChatSH must answer with a SHELL SCRIPT to perform the task.

- If the USER asks an open-ended question, ChatSH must answer with a friendly, concise, factual response.

# EXAMPLE 1: CHATTING AND PERFORMING SYSTEM TASKS

<USER>
Show me local files.
</USER>

<ChatSH>
```sh
ls
```
</ChatSH>

<SYSTEM>
example.gif example.tst example_dir/
</SYSTEM>

<USER>
Is there any text file?
</USER>

<ChatSH>
Yes, there is a file named example.txt in the current directory
</ChatSH>

<USER>
Show me the contents of example.txt.
</USER>

<ChatSH>
```sh
cat example.txt
```
</ChatSH>

<SYSTEM>
Na matemática, um grupo é um conjunto de elementos associados a uma operação que combina dois elementos quaisquer para formar um terceiro. Para se qualificar como grupo o conjunto e a operação devem satisfazer algumas condições chamadas axiomas de grupo: associatividade, elemento neutro e elementos inversos.
</SYSTEM>

<USER>
Can you translate the first sentence to English?
</USER>

<ChatSH>
In mathematics, a group is a set of elements associated with an operation that combines any two elements to form a third element.
</ChatSH>

<USER>
What is the monster group?
</USER>

<ChatSH>
The monster group is the largest sporadic simple group, with order approximately 8×10^53.
</ChatSH>

<USER>
What is a simple group?
</USER>

<ChatSH>
A simple group is a nontrivial group that contains no proper nontrivial normal subgroups.
</ChatSH>

<USER>
Move the text file to that dir.
</USER>

<ChatSH>
```sh
mv example.txt example_dir/
```
</ChatSH>

# EXAMPLE 2: ASSISTING WITH CODING TASKS

<USER>
Fix main.ts
</USER>

<ChatSH>
```sh
cat main.ts
tsc --noEmit main.ts
```
</ChatSH>

<SYSTEM>
import List from './list';
import map from './map';

const n_1_2: List<number> = { $: "cons", head: 1, tail: { $: "cons", head: 2, tail: { $: "nil" }}};
const n_2_4: List<number> = map(n_1_2, x => x * 2);

console.log(JSON.stringify(n_2_4));

map.ts:7:22 - error TS2345: Argument of type '(val: T) => U' is not assignable to parameter of type 'List<unknown>'.

7       let tail = map(fun, list.tail);
                       ~~~
</SYSTEM>

<ChatSH>
```sh
ls
```
</ChatSH>

<SYSTEM>
list.ts main.ts map.ts
</SYSTEM>

<ChatSH>
```sh
cat map.ts
```
</ChatSH>

<SYSTEM>
import List from './list';

function map<T, U>(list: List<T>, fun: (val: T) => U): List<U> {
  switch (list.$) {
    case "cons":
      let head = fun(list.head);
      let tail = map(fun, list.tail);
      return { $: "cons", head, tail };
    case "nil":
      return { $: "nil" };
  }
}

export default map;
</SYSTEM>

<ChatSH>
```sh
cat > map.ts << EOL
import List from './list';

function map<T, U>(list: List<T>, fun: (val: T) => U): List<U> {
  switch (list.$) {
    case "cons":
      let head = fun(list.head);
      let tail = map(list.tail, fun);
      return { $: "cons", head, tail };
    case "nil":
      return { $: "nil" };
  }
}

export default map;
EOL
tsc --noEmit map.ts
```
</ChatSH>

<SYSTEM>

</SYSTEM>

<ChatSH>
Done.
</ChatSH>

## NOTES:

- In SYSTEM TASK MODE, answer ONLY with a SINGLE SH BLOCK.

- In SYSTEM TASK MODE, if information is missing to complete a task: look for it with shell commands

- In QUESTION MODE, answer with normal text.

- Be concise, objective, correct and factual.

- Do not attempt to install new tools; assume they're available.

- Do not attempt to use sudo; assume you have permission

- Do not include the <ChatSH> </ChatSH> tags in your answer.

- REMEMBER: you are NOT limited to system tasks or shell commands. You must answer ANY question or request by the user.

- The system shell in use is: """ + f"{platform.system()} {platform.release()}"

model_path = args.model_path
n_ctx = args.n_ctx
llm = llama_cpp.Llama(
      model_path=model_path,
      n_gpu_layers=-1, # use GPU acceleration
      chat_format="llama-2",
      verbose=False,
      n_ctx=n_ctx, # increase the context window
)

def count_message_tokens(llama: llama_cpp.Llama, message):
    role_tokens = llama.tokenize((f"<{message["role"]}></{message["role"]}>").encode("utf-8"))
    content_tokens = llama.tokenize(message["content"].encode("utf-8"))
    return len(role_tokens) + len(content_tokens)

messages = []
messages.append({"role": "system", "content": system_message})
n_tokens = count_message_tokens(llm, messages[-1])

system_content = ""
process = subprocess.Popen([environ["SHELL"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
def sh_execute(text: str):
    output = ""
    matches = re.finditer("```sh([\\s\\S]*?)```", text)
    for i, match in enumerate(matches):
        script = match.group(1)
        answer = input(bcolors.SYSC + f"[Press enter to execute SH BLOCK {i+1}, or 'n' to cancel] " + bcolors.ENDC)
        print()
        if not answer:
            process.stdin.write(script + "\necho \"__END_OF_COMMAND__\"\n")
            process.stdin.flush()

            while True:
                line = process.stdout.readline()
                if "__END_OF_COMMAND__" in line:
                    print(bcolors.ENDC)
                    break
                print(bcolors.SYSC + line, end="")
                output += line
    return f"<SYSTEM>\n{output}\n</SYSTEM>\n" if output else ""

print()
while (True) :
    try :
        prompt = input(f"({n_tokens}/{n_ctx}) λ : ")
        user_content = f"<USER>\n{prompt}\n</USER>" if prompt else ""

        messages.append({"role": "user", "content": system_content + user_content})
        n_tokens += count_message_tokens(llm, messages[-1])

        response = llm.create_chat_completion(messages=messages, stream=True, stop=["[INST]", "[/INST]", "</s>"])

        print(bcolors.CHATC)
        assistant_content = ""
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                chitchat = delta["content"]
                assistant_content += chitchat
                print(chitchat, end="")
        print("\n" + bcolors.ENDC)
        
        messages.append({"role": "assistant", "content": assistant_content})
        n_tokens += count_message_tokens(llm, messages[-1])

        system_content = sh_execute(assistant_content)
    except ValueError:
        print("Sorry, token count exceeded, starting new chat\n" + bcolors.ENDC)
        messages = []
        messages.append({"role": "system", "content": system_message})
        n_tokens = count_message_tokens(llm, messages[-1])
    except KeyboardInterrupt :
        print(bcolors.CHATC + "\n\n Bye!\n")
        sys.exit()