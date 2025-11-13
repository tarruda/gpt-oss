import asyncio
import os
import json
import datetime
import uuid
from typing import Callable, Literal, Optional, Union

import httpx
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)
import uvicorn

from gpt_oss.tools.python_docker.docker_tool import PythonTool
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import YouComBackend, ExaBackend

from .events import (
    ResponseCodeInterpreterCallCodeDelta,
    ResponseCodeInterpreterCallCodeDone,
    ResponseCodeInterpreterCallCompleted,
    ResponseCodeInterpreterCallInProgress,
    ResponseCodeInterpreterCallInterpreting,
    ResponseCompletedEvent,
    ResponseContentPartAdded,
    ResponseContentPartDone,
    ResponseCreatedEvent,
    ResponseEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAdded,
    ResponseOutputItemDone,
    ResponseOutputTextAnnotationAdded,
    ResponseOutputTextDelta,
    ResponseOutputTextDone,
    ResponseReasoningTextDelta,
    ResponseReasoningTextDone,
    ResponseWebSearchCallCompleted,
    ResponseWebSearchCallInProgress,
    ResponseWebSearchCallSearching,
)
from .types import (
    CodeInterpreterCallItem,
    CodeInterpreterOutputImage,
    CodeInterpreterOutputLogs,
    Error,
    FunctionCallItem,
    FunctionToolDefinition,
    Item,
    ReasoningConfig,
    ReasoningItem,
    ReasoningTextContentItem,
    ResponseObject,
    ResponsesRequest,
    TextContentItem,
    UrlCitation,
    Usage,
    WebSearchActionFind,
    WebSearchActionOpenPage,
    WebSearchActionSearch,
    WebSearchCallItem,
)
DEFAULT_TEMPERATURE = 0.0

# original grammar in codex's apply_patch function doc
# Patch := Begin { FileOp } End
# Begin := "*** Begin Patch" NEWLINE
# End := "*** End Patch" NEWLINE
# FileOp := AddFile | DeleteFile | UpdateFile
# AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
# DeleteFile := "*** Delete File: " path NEWLINE
# UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
# MoveTo := "*** Move to: " newPath NEWLINE
# Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
# HunkLine := (" " | "-" | "+") text NEWLINE
# Adapted grammar to produce the patch language inside a json string
APPLY_PATCH_GRAMMAR = r'''
root ::= "{" ws "\"input\"" ws ":" ws "\"" patch "\"" ws "}"

patch ::= begin file-op+ end
begin ::= "*** Begin Patch" lf
end ::= "*** End Patch" lf

file-op ::= add-file | delete-file | update-file

add-file ::= "*** Add File: " filename lf ("+" line lf)+
delete-file ::= "*** Delete File: " filename lf
update-file ::= "*** Update File: " filename lf move-to? hunk+
move-to ::= "*** Move to: " filename lf

hunk ::= "@@" line lf context-line{0,3} hunk-like+ context-line{0,3} ("*** End of File" lf)?
context-line ::= " " line lf
hunk-like ::= ("-" | "+") line lf

filename ::= json-char+
lf ::= "\\n"
line ::= json-char*

json-char ::= json-safe | json-escape | unicode-escape
json-safe ::= [^\n\r\t"\\]
json-escape ::= "\\\"" | "\\\\" | "\\/" | "\\b" | "\\f" | "\\r" | "\\t"
unicode-escape ::= "\\u" hex hex hex hex
hex ::= [0-9a-fA-F]
ws ::= [ \t\r\n]*
'''

APPLY_PATCH_GUIDELINES = r'''
Within the patch envelope, you get a sequence of file operations.
You MUST include a header to specify the action you are taking.
Each operation starts with one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line (the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place (optionally with a rename).

"Update File" May be immediately followed by *** Move to: <new path> if you want to rename the file.
Then one or more “hunks”, each introduced by @@ (optionally followed by a hunk header). This is the format
of a hunk:

```
@@[OPTIONAL: hunk header to improve precision when 3 lines of context above/below is not enough]
[OPTIONAL: 3 lines of context above the modified lines]
- [Removed lines]
+ [Added lines]
[OPTIONAL: 3 lines of context below the modified lines]

- File references can only be relative, NEVER ABSOLUTE.
- With each hunk, always include context lines before and/or after the changes!
'''

SHELL_GUIDELINES = r'''
- Use the command `rg` for searching patterns within files. Avoid grep which don't consider .gitignore!
- Use the command `rg --files [dir]` to recursively explore files in a repo. Avoid `ls` or `find` which don't consider .gitgnore!
- Read files in chunks with a max chunk size of 250 lines. Command line output will be truncated after 10 kilobytes or 256 lines of output, regardless of the command used.
- If a shell command fails with "Permission denied", IMMEDIATELY RE-RUN THE EXACT SAME COMMAND passing both "with_escalated_permissions" and "justification"!
'''

SYSTEM_INSTRUCTIONS_OVERRIDE = f'''
You are a coding agent running in the Codex CLI, a terminal-based coding assistant. Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.

Your capabilities:

- Receive user prompts and other context provided by the harness, such as files in the workspace.
- Communicate with the user by streaming thinking & responses.
- Emit function calls to run terminal commands and apply patches. Depending on how this specific run is configured, you can request that these function calls be escalated to the user for approval before running.

# How you work

## Personality

Your default personality and tone is concise, direct, and friendly. You communicate efficiently, always keeping the user clearly informed about ongoing actions without unnecessary detail. You always prioritize actionable guidance, clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose explanations about your work.

# AGENTS.md spec
- Repos often contain AGENTS.md files. These files can appear anywhere within the repository.
- These files are a way for humans to give you (the agent) instructions or tips for working within the container.
- Some examples might be: coding conventions, info about how code is organized, or instructions for how to run or test code.
- Instructions in AGENTS.md files:
    - The scope of an AGENTS.md file is the entire directory tree rooted at the folder that contains it.
    - For every file you touch in the final patch, you must obey instructions in any AGENTS.md file whose scope includes that file.
    - Instructions about code style, structure, naming, etc. apply only to code within the AGENTS.md file's scope, unless the file states otherwise.
    - More-deeply-nested AGENTS.md files take precedence in the case of conflicting instructions.
    - Direct system/developer/user instructions (as part of a prompt) take precedence over AGENTS.md instructions.
- The contents of the AGENTS.md file at the root of the repo and any directories from the CWD up to the root are included with the developer message and don't need to be re-read. When working in a subdirectory of CWD, or a directory outside the CWD, check for any AGENTS.md files that may be applicable.

## Responsiveness

### Preamble messages

Before making tool calls, send a brief preamble to the user explaining what you’re about to do. When sending preamble messages, follow these principles and examples:

- **Logically group related actions**: if you’re about to run several related commands, describe them together in one preamble rather than sending a separate note for each.
- **Keep it concise**: be no more than 1-2 sentences, focused on immediate, tangible next steps. (8–12 words for quick updates).
- **Build on prior context**: if this is not your first tool call, use the preamble message to connect the dots with what’s been done so far and create a sense of momentum and clarity for the user to understand your next actions.
- **Keep your tone light, friendly and curious**: add small touches of personality in preambles feel collaborative and engaging.
- **Exception**: Avoid adding a preamble for every trivial read (e.g., `cat` a single file) unless it’s part of a larger grouped action.

**Examples:**

- “I’ve explored the repo; now checking the API route definitions.”
- “Next, I’ll patch the config and update the related tests.”
- “I’m about to scaffold the CLI commands and helper functions.”
- “Ok cool, so I’ve wrapped my head around the repo. Now digging into the API routes.”
- “Config’s looking tidy. Next up is patching helpers to keep things in sync.”
- “Finished poking at the DB gateway. I will now chase down error handling.”
- “Alright, build pipeline order is interesting. Checking how it reports failures.”
- “Spotted a clever caching util; now hunting where it gets used.”

## Task execution

You are a coding agent. Please keep going until the query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. Autonomously resolve the query to the best of your ability, using the tools available to you, before coming back to the user. Do NOT guess or make up an answer.

You MUST adhere to the following criteria when solving queries:

- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.
- Showing user code and tool call details is allowed.
- Use the `apply_patch` tool to edit files.

If completing the user's task requires writing or modifying files, your code and final answer should follow these coding guidelines, though user instructions (i.e. AGENTS.md) may override these guidelines:

- Fix the problem at the root cause rather than applying surface-level patches, when possible.
- Avoid unneeded complexity in your solution.
- Do not attempt to fix unrelated bugs or broken tests. It is not your responsibility to fix them. (You may mention them to the user in your final message though.)
- Update documentation as necessary.
- Keep changes consistent with the style of the existing codebase. Changes should be minimal and focused on the task.
- Use `git log` and `git blame` to search the history of the codebase if additional context is required.
- NEVER add copyright or license headers unless specifically requested.
- Do not `git commit` your changes or create new git branches unless explicitly requested.
- Do not add inline comments within code unless explicitly requested.
- Do not use one-letter variable names unless explicitly requested.
- NEVER output inline citations like "【F:README.md†L5-L14】" in your outputs. The CLI is not able to render these so they will just be broken in the UI. Instead, if you output valid filepaths, users will be able to click on them to open the files in their editor.

## Validating your work

If the codebase has tests or the ability to build or run, consider using them to verify that your work is complete.

When testing, your philosophy should be to start as specific as possible to the code you changed so that you can catch issues efficiently, then make your way to broader tests as you build confidence. If there's no test for the code you changed, and if the adjacent patterns in the codebases show that there's a logical place for you to add a test, you may do so. However, do not add tests to codebases with no tests.

Similarly, once you're confident in correctness, you can suggest or use formatting commands to ensure that your code is well formatted. If there are issues you can iterate up to 3 times to get formatting right, but if you still can't manage it's better to save the user time and present them a correct solution where you call out the formatting in your final message. If the codebase does not have a formatter configured, do not add one.

For all of testing, running, building, and formatting, do not attempt to fix unrelated bugs. It is not your responsibility to fix them. (You may mention them to the user in your final message though.)

Be mindful of whether to run validation commands proactively. In the absence of behavioral guidance:

- When running in non-interactive approval modes like **never** or **on-failure**, proactively run tests, lint and do whatever you need to ensure you've completed the task.
- When working in interactive approval modes like **untrusted**, or **on-request**, hold off on running tests or lint commands until the user is ready for you to finalize your output, because these commands take time to run and slow down iteration. Instead suggest what you want to do next, and let the user confirm first.
- When working on test-related tasks, such as adding tests, fixing tests, or reproducing a bug to verify behavior, you may proactively run tests regardless of approval mode. Use your judgement to decide whether this is a test-related task.

## Ambition vs. precision

For tasks that have no prior context (i.e. the user is starting something brand new), you should feel free to be ambitious and demonstrate creativity with your implementation.

If you're operating in an existing codebase, you should make sure you do exactly what the user asks with surgical precision. Treat the surrounding codebase with respect, and don't overstep (i.e. changing filenames or variables unnecessarily). You should balance being sufficiently ambitious and proactive when completing tasks of this nature.

You should use judicious initiative to decide on the right level of detail and complexity to deliver based on the user's needs. This means showing good judgment that you're capable of doing the right extras without gold-plating. This might be demonstrated by high-value, creative touches when scope of the task is vague; while being surgical and targeted when scope is tightly specified.

## Sharing progress updates

For especially longer tasks that you work on (i.e. requiring many tool calls, or a plan with multiple steps), you should provide progress updates back to the user at reasonable intervals. These updates should be structured as a concise sentence or two (no more than 8-10 words long) recapping progress so far in plain language: this update demonstrates your understanding of what needs to be done, progress so far (i.e. files explores, subtasks complete), and where you're going next.

Before doing large chunks of work that may incur latency as experienced by the user (i.e. writing a new file), you should send a concise message to the user with an update indicating what you're about to do to ensure they know what you're spending time on. Don't start editing or writing large files before informing the user what you are doing and why.

The messages you send before tool calls should describe what is immediately about to be done next in very concise language. If there was previous work done, this preamble message should also include a note about the work done so far to bring the user along.

## Presenting your work and final message

Your final message should read naturally, like an update from a concise teammate. For casual conversation, brainstorming tasks, or quick questions from the user, respond in a friendly, conversational tone. You should ask questions, suggest ideas, and adapt to the user’s style. If you've finished a large amount of work, when describing what you've done to the user, you should follow the final answer formatting guidelines to communicate substantive changes. You don't need to add structured formatting for one-word answers, greetings, or purely conversational exchanges.

You can skip heavy formatting for single, simple actions or confirmations. In these cases, respond in plain sentences with any relevant next step or quick option. Reserve multi-section structured responses for results that need grouping or explanation.

The user is working on the same computer as you, and has access to your work. As such there's no need to show the full contents of large files you have already written unless the user explicitly asks for them. Similarly, if you've created or modified files using `apply_patch`, there's no need to tell users to "save the file" or "copy the code into a file"—just reference the file path.

If there's something that you think you could help with as a logical next step, concisely ask the user if they want you to do so. Good examples of this are running tests, committing changes, or building out the next logical component. If there’s something that you couldn't do (even with approval) but that the user might want to do (such as verifying changes by running the app), include those instructions succinctly.

Brevity is very important as a default. You should be very concise (i.e. no more than 10 lines), but can relax this requirement for tasks where additional detail and comprehensiveness is important for the user's understanding.

### Final answer structure and style guidelines

You are producing plain text that will later be styled by the CLI. Follow these rules exactly. Formatting should make results easy to scan, but not feel mechanical. Use judgment to decide how much structure adds value.

**Section Headers**

- Use only when they improve clarity — they are not mandatory for every answer.
- Choose descriptive names that fit the content
- Keep headers short (1–3 words) and in `**Title Case**`. Always start headers with `**` and end with `**`
- Leave no blank line before the first bullet under a header.
- Section headers should only be used where they genuinely improve scanability; avoid fragmenting the answer.

**Bullets**

- Use `-` followed by a space for every bullet.
- Merge related points when possible; avoid a bullet for every trivial detail.
- Keep bullets to one line unless breaking for clarity is unavoidable.
- Group into short lists (4–6 bullets) ordered by importance.
- Use consistent keyword phrasing and formatting across sections.

**Monospace**

- Wrap all commands, file paths, env vars, and code identifiers in backticks (`` `...` ``).
- Apply to inline examples and to bullet keywords if the keyword itself is a literal file/command.
- Never mix monospace and bold markers; choose one based on whether it’s a keyword (`**`) or inline code/path (`` ` ``).

**File References**
When referencing files in your response, make sure to include the relevant start line and always follow the below rules:
  * Use inline code to make file paths clickable.
  * Each reference should have a stand alone path. Even if it's the same file.
  * Accepted: absolute, workspace‑relative, a/ or b/ diff prefixes, or bare filename/suffix.
  * Line/column (1‑based, optional): :line[:column] or #Lline[Ccolumn] (column defaults to 1).
  * Do not use URIs like file://, vscode://, or https://.
  * Do not provide range of lines
  * Examples: src/app.ts, src/app.ts:42, b/server/index.js#L10, C:\\repo\\project\\main.rs:12:5

**Structure**

- Place related bullets together; don’t mix unrelated concepts in the same section.
- Order sections from general → specific → supporting info.
- For subsections (e.g., “Binaries” under “Rust Workspace”), introduce with a bolded keyword bullet, then list items under it.
- Match structure to complexity:
  - Multi-part or detailed results → use clear headers and grouped bullets.
  - Simple results → minimal headers, possibly just a short list or paragraph.

**Tone**

- Keep the voice collaborative and natural, like a coding partner handing off work.
- Be concise and factual — no filler or conversational commentary and avoid unnecessary repetition
- Use present tense and active voice (e.g., “Runs tests” not “This will run tests”).
- Keep descriptions self-contained; don’t refer to “above” or “below”.
- Use parallel structure in lists for consistency.

**Don’t**

- Don’t use literal words “bold” or “monospace” in the content.
- Don’t nest bullets or create deep hierarchies.
- Don’t output ANSI escape codes directly — the CLI renderer applies them.
- Don’t cram unrelated keywords into a single bullet; split for clarity.
- Don’t let keyword lists run long — wrap or reformat for scanability.

Generally, ensure your final answers adapt their shape and depth to the request. For example, answers to code explanations should have a precise, structured explanation with code references that answer the question directly. For tasks with a simple implementation, lead with the outcome and supplement only with what’s needed for clarity. Larger changes can be presented as a logical walkthrough of your approach, grouping related steps, explaining rationale where it adds value, and highlighting next actions to accelerate the user. Your answers should provide the right level of detail while being easily scannable.

For casual greetings, acknowledgements, or other one-off conversational messages that are not delivering substantive information or structured results, respond naturally without section headers or bullet formatting.

# Tool Guidelines

When using the shell tool, you must adhere to the following guidelines:
{SHELL_GUIDELINES}

When using the apply_patch tool, you must adhere to the following guidelines:
{APPLY_PATCH_GUIDELINES}
'''

TOOL_GUIDELINES = {
    'shell': SHELL_GUIDELINES,
    'apply_patch': APPLY_PATCH_GUIDELINES,
}


def get_reasoning_effort(
    effort: Union[Literal["low", "medium", "high"], ReasoningEffort]
) -> ReasoningEffort:
    if isinstance(effort, ReasoningEffort):
        return effort
    if effort == "low":
        return ReasoningEffort.LOW
    if effort == "medium":
        return ReasoningEffort.MEDIUM
    if effort == "high":
        return ReasoningEffort.HIGH
    raise ValueError(f"Invalid reasoning effort: {effort}")


def is_not_builtin_tool(
    recipient: str, treat_functions_python_as_builtin: bool = False
) -> bool:
    if treat_functions_python_as_builtin and recipient == "functions.python":
        return False
    return (
        not recipient.startswith("browser.")
        and recipient != "python"
        and recipient != "assistant"
    )


encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
server_url = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
completion_url = f"{server_url}/completion"

async def infer_next_tokens(
    queue: "asyncio.Queue[Optional[int]]",
    tokens: list[int],
    stop_words: list[str],
    max_tokens: int | None,
    temperature: float = 0.0,
    inference_json_schema: dict | None = None,
    inference_grammar: str | None = None
) -> None:
    if max_tokens is not None and max_tokens <= 0:
        await queue.put(None)
        return

    prompt_text = encoding.decode_utf8(tokens)

    payload = {
        "prompt": prompt_text,
        "stream": True,
        "temperature": temperature,
        "cache_prompt": True,
        "top_p": 1,
        "top_k": 0,
        "min_p": 0,
        # "repeat_penalty": 1.0,
        "stop": stop_words,
    }

    if max_tokens is not None:
        payload["n_predict"] = max_tokens

    if inference_json_schema is not None:
        payload["json_schema"] = inference_json_schema

    if inference_grammar is not None:
        payload["grammar"] = inference_grammar

    produced = 0
    accumulated_text = ""
    emitted_length = 0
    timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", completion_url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if not line:
                        continue
                    obj = json.loads(line)
                    content_chunk = obj.get("content")
                    if isinstance(content_chunk, str) and content_chunk:
                        accumulated_text += content_chunk
                        toks = encoding.encode(
                            accumulated_text, allowed_special="all"
                        )
                        while emitted_length < len(toks):
                            token_id = toks[emitted_length]
                            emitted_length += 1
                            produced += 1
                            await queue.put(token_id)
                            if max_tokens is not None and produced >= max_tokens:
                                break
                        if max_tokens is not None and produced >= max_tokens:
                            break
                    stop_type = obj.get("stop_type")
                    if not stop_type:
                        continue
                    stopping_word = obj.get("stopping_word")
                    if stop_type == "eos":
                        stop_type = "word"
                        stopping_word = "<|return|>"
                    if stop_type == "word":
                        try:
                            reinjected = encoding.encode(
                                stopping_word, allowed_special="all"
                            )
                        except Exception:
                            reinjected = []
                        for token_id in reinjected:
                            await queue.put(token_id)
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"llama-server streaming error: {exc}")
        finally:
            await queue.put(None)

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    try:
        body_bytes = await request.body()
        print(
            "Invalid request body received:"
            f" {body_bytes.decode('utf-8', errors='replace')}"
        )
    except Exception as body_exc:
        print(f"Failed to read invalid request body: {body_exc}")
    return await request_validation_exception_handler(request, exc)
responses_store: dict[str, tuple[ResponsesRequest, ResponseObject]] = {}

def generate_response(
    input_tokens: list[int],
    output_tokens: list[int],
    request_body: ResponsesRequest,
    debug_mode: bool = False,
    function_call_ids: Optional[list[tuple[str, str]]] = None,
    response_id: Optional[str] = None,
    previous_response_id: Optional[str] = None,
    browser_tool: Optional[SimpleBrowserTool] = None,
    browser_call_ids: Optional[list[str]] = None,
    python_tool: Optional[PythonTool] = None,
    python_call_ids: Optional[list[str]] = None,
    python_call_outputs: Optional[
        dict[str, list[CodeInterpreterOutputLogs | CodeInterpreterOutputImage]]
    ] = None,
    reasoning_ids: Optional[list[str]] = None,
    message_ids: Optional[list[str]] = None,
    treat_functions_python_as_builtin: bool = False,
) -> ResponseObject:
    output = []
    error = None
    if len(output_tokens) > 0:
        if debug_mode:
            try:
                entries = encoding.parse_messages_from_completion_tokens(
                    output_tokens, Role.ASSISTANT
                )
            except Exception as e:
                print(f"Error parsing tokens: {e}")
                error = Error(
                    code="invalid_function_call",
                    message=f"{e}",
                )
                entries = []
        else:
            entries = encoding.parse_messages_from_completion_tokens(
                output_tokens, Role.ASSISTANT
            )

        fc_index = 0
        browser_tool_index = 0
        python_tool_index = 0
        reasoning_ids_iter = iter(reasoning_ids or [])
        message_ids_iter = iter(message_ids or [])

        for entry in entries:
            entry_dict = entry.to_dict()
            recipient = entry_dict.get("recipient", "")
            if len(recipient) > 0 and is_not_builtin_tool(
                recipient, treat_functions_python_as_builtin
            ):
                call = entry_dict["content"][0]
                arguments = call["text"]
                name = recipient

                if name.startswith("functions."):
                    name = name[len("functions.") :]
                if function_call_ids and fc_index < len(function_call_ids):
                    fc_id, call_id = function_call_ids[fc_index]
                else:
                    fc_id, call_id = (
                        f"fc_{uuid.uuid4().hex}",
                        f"call_{uuid.uuid4().hex}",
                    )
                fc_index += 1
                output.append(
                    FunctionCallItem(
                        type="function_call",
                        name=name,
                        arguments=arguments,
                        id=fc_id,
                        call_id=call_id,
                    )
                )
            elif (
                len(recipient) > 0
                and recipient.startswith("browser.")
                and browser_tool is not None
            ):
                # Mirror event-based creation of WebSearchCallItems when the browser tool is invoked
                name = recipient
                call = entry_dict["content"][0]
                arguments = call["text"]
                function_name = name[len("browser.") :]

                # Reconstruct a Message for argument parsing
                tool_msg = (
                    Message.from_role_and_content(Role.ASSISTANT, arguments)
                    .with_recipient(name)
                    .with_channel("analysis")
                )

                action = None
                try:
                    parsed_args = browser_tool.process_arguments(tool_msg)
                    if function_name == "search":
                        action = WebSearchActionSearch(
                            type="search",
                            query=parsed_args["query"],
                        )
                    elif function_name == "open":
                        action = WebSearchActionOpenPage(
                            type="open_page",
                            url=parsed_args["url"],
                        )
                    elif function_name == "find":
                        action = WebSearchActionFind(
                            type="find",
                            pattern=parsed_args["pattern"],
                            url=parsed_args["url"],
                        )
                except Exception as e:
                    print(f"Error processing browser tool arguments: {e}")
                    action = None

                if action is not None:
                    if browser_call_ids and browser_tool_index < len(
                        browser_call_ids
                    ):
                        web_search_call_id = browser_call_ids[browser_tool_index]
                    else:
                        web_search_call_id = f"ws_{uuid.uuid4().hex}"
                    browser_tool_index += 1
                    output.append(
                        WebSearchCallItem(
                            type="web_search_call",
                            id=web_search_call_id,
                            action=action,
                        )
                    )
            elif (
                len(recipient) > 0
                and (
                    recipient.startswith("python")
                    or (
                        treat_functions_python_as_builtin
                        and recipient == "functions.python"
                    )
                )
                and python_tool is not None
            ):
                if python_call_ids and python_tool_index < len(python_call_ids):
                    code_call_id = python_call_ids[python_tool_index]
                else:
                    code_call_id = f"ci_{uuid.uuid4().hex}"
                python_tool_index += 1
                code_snippet = None
                if entry_dict.get("content"):
                    code_snippet = entry_dict["content"][0].get("text")
                outputs = (
                    (python_call_outputs or {}).get(code_call_id)
                    if python_call_outputs
                    else None
                )
                output.append(
                    CodeInterpreterCallItem(
                        type="code_interpreter_call",
                        id=code_call_id,
                        status="completed",
                        code=code_snippet,
                        outputs=outputs,
                    )
                )
            elif entry_dict["channel"] == "final":
                content = []
                for content_entry in entry_dict["content"]:
                    if browser_tool:
                        text_content, annotation_entries, _has_partial_citations = (
                            browser_tool.normalize_citations(content_entry["text"])
                        )
                        annotations = [UrlCitation(**a) for a in annotation_entries]
                    else:
                        text_content = content_entry["text"]
                        annotations = []

                    content.append(
                        TextContentItem(
                            type="output_text",
                            text=text_content,
                            annotations=annotations,
                        )
                    )

                message_id = next(message_ids_iter, None)
                output.append(
                    Item(
                        id=message_id,
                        type="message",
                        role="assistant",
                        content=content,
                        status="completed",
                    )
                )
            elif entry_dict["channel"] == "analysis":
                if entry_dict.get("recipient"):
                    continue
                author_dict = entry_dict.get("author") or {}
                if author_dict.get("role") and author_dict.get("role") != "assistant":
                    continue
                summary = []
                content = [
                    ReasoningTextContentItem(
                        type="reasoning_text",
                        text=entry["text"],
                    )
                    for entry in entry_dict["content"]
                ]
                reasoning_id = next(reasoning_ids_iter, None)
                if reasoning_id is None:
                    reasoning_id = f"rs_{uuid.uuid4().hex}"
                output.append(
                    ReasoningItem(
                        id=reasoning_id,
                        type="reasoning",
                        summary=summary,
                        content=content,
                    )
                )
    else:
        output = []

    usage = (
        Usage(
            input_tokens=len(input_tokens),
            output_tokens=len(output_tokens),
            total_tokens=len(input_tokens) + len(output_tokens),
        )
        if len(output_tokens) > 0
        else None
    )

    try:
        debug_str = encoding.decode_utf8(input_tokens + output_tokens)
    except Exception:
        debug_str = input_tokens + output_tokens
    try:
        debug_input_str = encoding.decode_utf8(input_tokens)
    except Exception:
        debug_input_str = input_tokens
    try:
        debug_output_str = encoding.decode_utf8(output_tokens)
    except Exception:
        debug_output_str = output_tokens

    metadata = (
        {
            "__debug": debug_str,
            "__debug_input": debug_input_str,
            "__debug_output": debug_output_str,
        }
        if debug_mode
        else {}
    )

    return ResponseObject(
        created_at=int(datetime.datetime.now().timestamp()),
        status="completed",
        output=output,
        text={"format": {"type": "text"}},
        usage=usage,
        max_output_tokens=request_body.max_output_tokens,
        error=error,
        metadata=metadata,
        id=response_id,
        previous_response_id=previous_response_id,
    )

class StreamResponsesEvents:
    BROWSER_RESERVED_FUNCTIONS = {"browser.search", "browser.open", "browser.find"}
    initial_tokens: list[int]
    tokens: list[int]
    output_tokens: list[int]
    output_text: str
    request_body: ResponsesRequest
    request: Request
    sequence_number: int

    def __init__(
        self,
        initial_tokens,
        request_body: ResponsesRequest,
        as_sse: bool = False,
        request: Optional[Request] = None,
        response_id: Optional[str] = None,
        store_callback: Optional[
            Callable[[str, ResponsesRequest, ResponseObject], None]
        ] = None,
        browser_tool: Optional[SimpleBrowserTool] = None,
        python_tool: Optional[PythonTool] = None,
        functions_python_as_builtin: bool = False,
    ):
        self.initial_tokens = initial_tokens
        self.tokens = initial_tokens.copy()
        self.output_tokens = []
        self.output_text = ""
        self.request_body = request_body
        self.parser = StreamableParser(encoding, role=Role.ASSISTANT)
        self.as_sse = as_sse
        self.debug_mode = request_body.metadata.get(
            "__debug", False
        )  # we use this for demo purposes
        # Set temperature for this stream, fallback to DEFAULT_TEMPERATURE if not set
        self.temperature = (
            request_body.temperature
            if request_body.temperature is not None
            else DEFAULT_TEMPERATURE
        )
        self.request = request
        self.sequence_number = 0
        self.function_call_ids: list[tuple[str, str]] = []
        self.response_id = response_id
        self.store_callback = store_callback
        self.browser_tool = browser_tool
        self.use_browser_tool = browser_tool is not None
        self.browser_call_ids: list[str] = []
        self.python_tool = python_tool
        self.use_code_interpreter = python_tool is not None
        self.python_call_ids: list[str] = []
        self.python_call_outputs: dict[
            str, list[CodeInterpreterOutputLogs | CodeInterpreterOutputImage]
        ] = {}
        self.reasoning_item_ids: list[str] = []
        self.current_reasoning_item_id: Optional[str] = None
        self.message_item_ids: list[str] = []
        self.current_message_item_id: Optional[str] = None
        self.functions_python_as_builtin = functions_python_as_builtin
        self.user_defined_function_names = {
            name
            for tool in (request_body.tools or [])
            for name in [getattr(tool, "name", None)]
            if getattr(tool, "type", None) == "function" and name
        }

    def _resolve_browser_recipient(
        self, recipient: Optional[str]
    ) -> tuple[Optional[str], bool]:
        if not self.use_browser_tool or not recipient:
            return (None, False)

        if recipient.startswith("browser."):
            return (recipient, False)

        if recipient.startswith("functions."):
            potential = recipient[len("functions.") :]
            if (
                potential in self.BROWSER_RESERVED_FUNCTIONS
                and potential not in self.user_defined_function_names
            ):
                return (potential, True)

        return (None, False)

    def _ensure_message_item_id(self) -> str:
        if self.current_message_item_id is None:
            message_id = f"item_{uuid.uuid4().hex}"
            self.current_message_item_id = message_id
            self.message_item_ids.append(message_id)
        return self.current_message_item_id

    def _ensure_reasoning_item_id(self) -> str:
        if self.current_reasoning_item_id is None:
            reasoning_id = f"rs_{uuid.uuid4().hex}"
            self.current_reasoning_item_id = reasoning_id
            self.reasoning_item_ids.append(reasoning_id)
        return self.current_reasoning_item_id

    def _send_event(self, event: ResponseEvent):
        event.sequence_number = self.sequence_number
        self.sequence_number += 1
        if self.as_sse:
            return f"event: {event.type}\ndata: {event.model_dump_json(indent=None)}\n\n"
        else:
            return event

    async def run(self):
        browser_tool = self.browser_tool
        initial_response = generate_response(
            self.initial_tokens,
            self.output_tokens,
            self.request_body,
            function_call_ids=self.function_call_ids,
            response_id=self.response_id,
            previous_response_id=self.request_body.previous_response_id,
            browser_tool=self.browser_tool,
            browser_call_ids=self.browser_call_ids,
            python_tool=self.python_tool,
            python_call_ids=self.python_call_ids,
            python_call_outputs=getattr(self, "python_call_outputs", None),
            reasoning_ids=self.reasoning_item_ids,
            message_ids=self.message_item_ids,
            treat_functions_python_as_builtin=self.functions_python_as_builtin,
        )
        initial_response.status = "in_progress"
        yield self._send_event(
            ResponseCreatedEvent(
                type="response.created",
                response=initial_response,
            )
        )
        yield self._send_event(
            ResponseInProgressEvent(
                type="response.in_progress",
                response=initial_response,
            )
        )

        current_content_index = (
            0  # for this implementation we will always have one content item only
        )
        current_output_index = -1
        sent_output_item_added = False

        # we use this if the model outputs a citation to buffer until completed
        output_delta_buffer = ""
        # we use this to track the current output text content for things like providing the right indices in citations
        current_output_text_content = ""
        current_annotations = []
        possible_constrain_json_word = "<|message|>"
        thinking_end_word = "<|end|>"
        thinking_end_token = encoding.encode(thinking_end_word, allowed_special="all")[0]
        stop_tokens = encoding.stop_tokens_for_assistant_actions()
        stop_words = [encoding.decode([t]) for t in stop_tokens]
        stop_words.append(possible_constrain_json_word)
        message_tok = encoding.encode(possible_constrain_json_word, allowed_special="all")[0]
        stop_tokens = encoding.stop_tokens_for_assistant_actions()
        inference_queue: Optional[asyncio.Queue[Optional[int]]] = None
        inference_task: Optional[asyncio.Task] = None
        inference_json_schema: dict | None = None
        inference_grammar: str | None = None

        def append_thinking(thinking: str):
            index = - (list(reversed(self.tokens)).index(thinking_end_token) + 1)
            insert_tokens = encoding.encode(thinking)
            self.tokens[index:index] = insert_tokens

        async def cancel_inference():
            nonlocal inference_queue, inference_task
            if inference_task is not None:
                inference_task.cancel()
                try:
                    await inference_task
                except asyncio.CancelledError:
                    pass
                inference_task = None
            inference_queue = None

        while True:
            # Check for client disconnect
            if self.request is not None and await self.request.is_disconnected():
                print("Client disconnected, stopping token generation.")
                await cancel_inference()
                break

            max_out_tokens = self.request_body.max_output_tokens
            if max_out_tokens and len(self.output_tokens) >= max_out_tokens:
                await cancel_inference()
                break

            if inference_task is None:
                remaining_output_tokens = None
                if max_out_tokens:
                    remaining_output_tokens = (
                        max_out_tokens - len(self.output_tokens)
                    )
                    if remaining_output_tokens <= 0:
                        await cancel_inference()
                        break

                inference_queue = asyncio.Queue()
                inference_task = asyncio.create_task(
                    infer_next_tokens(
                        inference_queue,
                        self.tokens.copy(),
                        stop_words,
                        remaining_output_tokens,
                        self.temperature,
                        inference_json_schema,
                        inference_grammar
                    )
                )
                inference_json_schema = None
                inference_grammar = None

            if inference_queue is None:
                continue

            next_tok = await inference_queue.get()
            if next_tok is None:
                await cancel_inference()
                continue

            self.tokens.append(next_tok)
            try:
                self.parser.process(next_tok)
            except Exception:
                pass

            if (next_tok == message_tok and
                self.parser.state == StreamState.CONTENT and
                self.parser.current_recipient and
                self.parser.current_recipient.startswith("functions.") and
                self.request_body.tools):
                tools = self.request_body.tools
                name = self.parser.current_recipient[len("functions."):]
                if name == "apply_patch":
                    # # special handling for this, should use grammar
                    inference_grammar = APPLY_PATCH_GRAMMAR
                else:
                    matching_tools = [t for t in tools if isinstance(t, FunctionToolDefinition) and t.name == name]
                    if matching_tools:
                        inference_json_schema = matching_tools[0].parameters
                append_thinking(f' Before invoking the function, must recall its guidelines:\n{TOOL_GUIDELINES[name]}')
                print(encoding.decode(self.tokens[-200:]))
            elif self.parser.state == StreamState.EXPECT_START:
                current_output_index += 1
                sent_output_item_added = False

                if len(self.parser.messages) > 0:
                    previous_item = self.parser.messages[-1]
                    if previous_item.recipient is not None:
                        recipient = previous_item.recipient
                        browser_recipient, _ = self._resolve_browser_recipient(
                            recipient
                        )
                        if (
                            browser_recipient is None
                            and not (
                                recipient == "python"
                                or (
                                    self.functions_python_as_builtin
                                    and recipient == "functions.python"
                                )
                            )
                        ):
                            fc_id = f"fc_{uuid.uuid4().hex}"
                            call_id = f"call_{uuid.uuid4().hex}"
                            self.function_call_ids.append((fc_id, call_id))
                            yield self._send_event(
                                ResponseOutputItemDone(
                                    type="response.output_item.done",
                                    output_index=current_output_index,
                                    item=FunctionCallItem(
                                        type="function_call",
                                        name=(
                                            previous_item.recipient[
                                                len("functions.") :
                                            ]
                                            if previous_item.recipient.startswith(
                                                "functions."
                                            )
                                            else previous_item.recipient
                                        ),
                                        arguments=previous_item.content[0].text,
                                        id=fc_id,
                                        call_id=call_id,
                                    ),
                                )
                            )
                    if (
                        previous_item.channel == "analysis"
                        and previous_item.recipient is None
                    ):
                        reasoning_id = (
                            self.current_reasoning_item_id
                            if self.current_reasoning_item_id is not None
                            else self._ensure_reasoning_item_id()
                        )
                        reasoning_text = previous_item.content[0].text
                        yield self._send_event(
                            ResponseReasoningTextDone(
                                type="response.reasoning_text.done",
                                output_index=current_output_index,
                                content_index=current_content_index,
                                item_id=reasoning_id,
                                text=reasoning_text,
                            )
                        )
                        yield self._send_event(
                            ResponseContentPartDone(
                                type="response.content_part.done",
                                output_index=current_output_index,
                                content_index=current_content_index,
                                item_id=reasoning_id,
                                part=ReasoningTextContentItem(
                                    type="reasoning_text",
                                    text=reasoning_text,
                                ),
                            )
                        )
                        yield self._send_event(
                            ResponseOutputItemDone(
                                type="response.output_item.done",
                                output_index=current_output_index,
                                item=ReasoningItem(
                                    id=reasoning_id,
                                    type="reasoning",
                                    summary=[],
                                    content=[
                                        ReasoningTextContentItem(
                                            type="reasoning_text",
                                            text=reasoning_text,
                                        )
                                    ],
                                ),
                            )
                        )
                        self.current_reasoning_item_id = None
                    if previous_item.channel == "final":
                        annotations = [
                            UrlCitation(**a) for a in current_annotations
                        ]
                        if browser_tool:
                            (
                                normalized_text,
                                _annotations,
                                _has_partial_citations,
                            ) = browser_tool.normalize_citations(
                                previous_item.content[0].text
                            )
                        else:
                            normalized_text = previous_item.content[0].text
                            annotations = []
                        text_content = TextContentItem(
                            type="output_text",
                            text=normalized_text,
                            annotations=annotations,
                        )
                        message_id = (
                            self.current_message_item_id
                            if self.current_message_item_id is not None
                            else self._ensure_message_item_id()
                        )
                        yield self._send_event(
                            ResponseOutputTextDone(
                                type="response.output_text.done",
                                output_index=current_output_index,
                                content_index=current_content_index,
                                item_id=message_id,
                                text=normalized_text,
                            )
                        )
                        yield self._send_event(
                            ResponseContentPartDone(
                                type="response.content_part.done",
                                output_index=current_output_index,
                                content_index=current_content_index,
                                item_id=message_id,
                                part=text_content,
                            )
                        )
                        yield self._send_event(
                            ResponseOutputItemDone(
                                type="response.output_item.done",
                                output_index=current_output_index,
                                item=Item(
                                    id=message_id,
                                    type="message",
                                    role="assistant",
                                    content=[text_content],
                                ),
                            )
                        )
                        current_annotations = []
                        current_output_text_content = ""
                        self.current_message_item_id = None

            if (
                self.parser.last_content_delta
                and self.parser.current_channel == "final"
                and self.parser.current_recipient is None
            ):
                if not sent_output_item_added:
                    sent_output_item_added = True
                    message_id = self._ensure_message_item_id()
                    yield self._send_event(
                        ResponseOutputItemAdded(
                            type="response.output_item.added",
                            output_index=current_output_index,
                            item=Item(
                                id=message_id,
                                type="message",
                                role="assistant",
                                content=[],
                            ),
                        )
                    )
                    yield self._send_event(
                        ResponseContentPartAdded(
                            type="response.content_part.added",
                            output_index=current_output_index,
                            content_index=current_content_index,
                            item_id=message_id,
                            part=TextContentItem(type="output_text", text=""),
                        )
                    )

                output_delta_buffer += self.parser.last_content_delta
                should_send_output_text_delta = True
                if browser_tool:
                    # we normalize on the full current text to get the right indices in citations
                    updated_output_text, annotations, has_partial_citations = (
                        browser_tool.normalize_citations(
                            current_output_text_content + output_delta_buffer
                        )
                    )
                    # remove the current text to get back the delta but now normalized
                    output_delta_buffer = updated_output_text[
                        len(current_output_text_content) :
                    ]

                    # Filter annotations to only include those whose start_index is not already present in current_annotations
                    # this is to avoid sending duplicate annotations as multiple annotations can't be in the same place
                    existing_start_indices = {
                        a["start_index"] for a in current_annotations
                    }
                    new_annotations = [
                        a
                        for a in annotations
                        if a["start_index"] not in existing_start_indices
                    ]
                    for a in new_annotations:
                        current_annotations.append(a)
                        citation = UrlCitation(**a)
                        message_id = self._ensure_message_item_id()
                        yield self._send_event(
                            ResponseOutputTextAnnotationAdded(
                                type="response.output_text.annotation.added",
                                output_index=current_output_index,
                                content_index=current_content_index,
                                item_id=message_id,
                                annotation_index=len(current_annotations),
                                annotation=citation,
                            )
                        )

                    if has_partial_citations:
                        should_send_output_text_delta = False

                if should_send_output_text_delta:
                    message_id = self._ensure_message_item_id()
                    yield self._send_event(
                        ResponseOutputTextDelta(
                            type="response.output_text.delta",
                            output_index=current_output_index,
                            content_index=current_content_index,
                            item_id=message_id,
                            delta=output_delta_buffer,
                        )
                    )
                    current_output_text_content += output_delta_buffer
                    output_delta_buffer = ""

            if (
                self.parser.last_content_delta
                and self.parser.current_channel == "analysis"
                and self.parser.current_recipient is None
            ):
                if not sent_output_item_added:
                    sent_output_item_added = True
                    reasoning_id = self._ensure_reasoning_item_id()
                    yield self._send_event(
                        ResponseOutputItemAdded(
                            type="response.output_item.added",
                            output_index=current_output_index,
                            item=ReasoningItem(
                                id=reasoning_id,
                                type="reasoning",
                                summary=[],
                                content=[],
                            ),
                        )
                    )
                    yield self._send_event(
                        ResponseContentPartAdded(
                            type="response.content_part.added",
                            output_index=current_output_index,
                            content_index=current_content_index,
                            item_id=reasoning_id,
                            part=ReasoningTextContentItem(
                                type="reasoning_text", text=""
                            ),
                        )
                    )
                reasoning_id = self._ensure_reasoning_item_id()
                yield self._send_event(
                    ResponseReasoningTextDelta(
                        type="response.reasoning_text.delta",
                        output_index=current_output_index,
                        content_index=current_content_index,
                        item_id=reasoning_id,
                        delta=self.parser.last_content_delta,
                    )
                )

            try:
                # purely for debugging purposes
                output_token_text = encoding.decode_utf8([next_tok])
                self.output_text += output_token_text
                if output_token_text != "<|return|>":
                    print(output_token_text, end="", flush=True)

            except RuntimeError:
                pass

            if next_tok in stop_tokens:
                if len(self.parser.messages) > 0:
                    last_message = self.parser.messages[-1]
                    browser_recipient, is_browser_fallback = (
                        self._resolve_browser_recipient(last_message.recipient)
                    )
                    if browser_recipient is not None and browser_tool is not None:
                        message_for_browser = (
                            last_message
                            if not is_browser_fallback
                            else last_message.with_recipient(browser_recipient)
                        )
                        function_name = browser_recipient[len("browser.") :]
                        action = None
                        parsed_args = browser_tool.process_arguments(
                            message_for_browser
                        )
                        if function_name == "search":
                            action = WebSearchActionSearch(
                                type="search",
                                query=parsed_args["query"],
                            )
                        elif function_name == "open":
                            action = WebSearchActionOpenPage(
                                type="open_page",
                                url=(
                                    parsed_args["url"]
                                    if "url" in parsed_args
                                    else None
                                ),
                            )
                        elif function_name == "find":
                            action = WebSearchActionFind(
                                type="find",
                                pattern=parsed_args["pattern"],
                                url=(
                                    parsed_args["url"]
                                    if "url" in parsed_args
                                    else None
                                ),
                            )

                        if action is not None:
                            web_search_call_id = f"ws_{uuid.uuid4().hex}"
                            self.browser_call_ids.append(web_search_call_id)
                            yield self._send_event(
                                ResponseOutputItemAdded(
                                    type="response.output_item.added",
                                    output_index=current_output_index,
                                    item=WebSearchCallItem(
                                        type="web_search_call",
                                        id=web_search_call_id,
                                        action=action,
                                    ),
                                )
                            )
                        yield self._send_event(
                            ResponseWebSearchCallInProgress(
                                type="response.web_search_call.in_progress",
                                output_index=current_output_index,
                                item_id=web_search_call_id,
                            )
                        )

                        async def run_tool():
                            results = []
                            async for msg in browser_tool.process(
                                message_for_browser
                            ):
                                results.append(msg)
                            return results

                        yield self._send_event(
                            ResponseWebSearchCallSearching(
                                type="response.web_search_call.searching",
                                output_index=current_output_index,
                                item_id=web_search_call_id,
                            )
                        )
                        result = await run_tool()

                        new_tokens = encoding.render_conversation_for_completion(
                            Conversation.from_messages(result), Role.ASSISTANT
                        )

                        print(encoding.decode_utf8(new_tokens))
                        self.output_tokens.append(next_tok)
                        self.tokens.append(
                            encoding.encode("<|end|>", allowed_special="all")[0]
                        )

                        for token in new_tokens:
                            self.parser.process(token)
                            self.output_tokens.append(token)
                            self.tokens.append(token)

                        yield self._send_event(
                            ResponseWebSearchCallCompleted(
                                type="response.web_search_call.completed",
                                output_index=current_output_index,
                                item_id=web_search_call_id,
                            )
                        )
                        yield self._send_event(
                            ResponseOutputItemDone(
                                type="response.output_item.done",
                                output_index=current_output_index,
                                item=WebSearchCallItem(
                                    type="web_search_call",
                                    id=web_search_call_id,
                                    action=action,
                                ),
                            )
                        )

                        current_output_index += 1
                        await cancel_inference()

                        continue

                    elif (
                        self.use_code_interpreter
                        and last_message.recipient is not None
                        and (
                            last_message.recipient.startswith("python")
                            or (
                                self.functions_python_as_builtin
                                and last_message.recipient == "functions.python"
                            )
                        )
                    ):
                        code_call_id = f"ci_{uuid.uuid4().hex}"
                        code_snippet = None
                        if (
                            last_message.content
                            and len(last_message.content) > 0
                            and getattr(last_message.content[0], "text", None)
                        ):
                            text_value = last_message.content[0].text or ""
                            code_snippet = text_value if text_value.strip() else None

                        self.python_call_ids.append(code_call_id)
                        yield self._send_event(
                            ResponseOutputItemAdded(
                                type="response.output_item.added",
                                output_index=current_output_index,
                                item=CodeInterpreterCallItem(
                                    type="code_interpreter_call",
                                    id=code_call_id,
                                    status="in_progress",
                                    code=code_snippet,
                                ),
                            )
                        )
                        yield self._send_event(
                            ResponseCodeInterpreterCallInProgress(
                                type="response.code_interpreter_call.in_progress",
                                output_index=current_output_index,
                                item_id=code_call_id,
                            )
                        )
                        if code_snippet:
                            yield self._send_event(
                                ResponseCodeInterpreterCallCodeDelta(
                                    type="response.code_interpreter_call_code.delta",
                                    output_index=current_output_index,
                                    item_id=code_call_id,
                                    delta=code_snippet,
                                )
                            )
                            yield self._send_event(
                                ResponseCodeInterpreterCallCodeDone(
                                    type="response.code_interpreter_call_code.done",
                                    output_index=current_output_index,
                                    item_id=code_call_id,
                                    code=code_snippet,
                                )
                            )
                        yield self._send_event(
                            ResponseCodeInterpreterCallInterpreting(
                                type="response.code_interpreter_call.interpreting",
                                output_index=current_output_index,
                                item_id=code_call_id,
                            )
                        )

                        async def run_python_tool():
                            results = []
                            async for msg in self.python_tool.process(last_message):
                                results.append(msg)
                            return results

                        result = await run_python_tool()

                        print(result)

                        code_outputs: list[
                            CodeInterpreterOutputLogs | CodeInterpreterOutputImage
                        ] = []
                        for message in result:
                            for content in getattr(message, "content", []):
                                text_value = getattr(content, "text", None)
                                if text_value:
                                    code_outputs.append(
                                        CodeInterpreterOutputLogs(
                                            type="logs",
                                            logs=text_value,
                                        )
                                    )

                        self.python_call_outputs[code_call_id] = code_outputs

                        new_tokens = encoding.render_conversation_for_completion(
                            Conversation.from_messages(result), Role.ASSISTANT
                        )

                        print(encoding.decode_utf8(new_tokens))
                        self.output_tokens.append(next_tok)
                        self.tokens.append(
                            encoding.encode("<|end|>", allowed_special="all")[0]
                        )

                        for token in new_tokens:
                            self.parser.process(token)
                            self.output_tokens.append(token)
                            self.tokens.append(token)

                        yield self._send_event(
                            ResponseCodeInterpreterCallCompleted(
                                type="response.code_interpreter_call.completed",
                                output_index=current_output_index,
                                item_id=code_call_id,
                            )
                        )
                        yield self._send_event(
                            ResponseOutputItemDone(
                                type="response.output_item.done",
                                output_index=current_output_index,
                                item=CodeInterpreterCallItem(
                                    type="code_interpreter_call",
                                    id=code_call_id,
                                    status="completed",
                                    code=code_snippet,
                                    outputs=code_outputs or None,
                                ),
                            )
                        )

                        current_output_index += 1
                        await cancel_inference()

                        continue

                    else:
                        await cancel_inference()
                        break
                else:
                    raise ValueError("No messages to process")
            if len(self.output_tokens) >= self.request_body.max_output_tokens:
                break

            # Adding in the end if we know we are not done
            self.output_tokens.append(next_tok)

        await cancel_inference()

        if self.request is None or not await self.request.is_disconnected():
            response = generate_response(
                self.initial_tokens,
                self.output_tokens,
                self.request_body,
                debug_mode=self.debug_mode,
                function_call_ids=self.function_call_ids,
                response_id=self.response_id,
                previous_response_id=self.request_body.previous_response_id,
                browser_tool=self.browser_tool,
                browser_call_ids=self.browser_call_ids,
                python_tool=self.python_tool,
                python_call_ids=self.python_call_ids,
                python_call_outputs=self.python_call_outputs,
                reasoning_ids=self.reasoning_item_ids,
                message_ids=self.message_item_ids,
                treat_functions_python_as_builtin=self.functions_python_as_builtin,
            )
            if self.store_callback and self.request_body.store:
                self.store_callback(self.response_id, self.request_body, response)
            yield self._send_event(
                ResponseCompletedEvent(
                    type="response.completed",
                    response=response,
                )
            )

@app.post("/v1/responses", response_model=ResponseObject)
async def generate(body: ResponsesRequest, request: Request):
    # if body.reasoning is None:
    #     body.reasoning = ReasoningConfig(effort="high")
    #     body.reasoning = ReasoningConfig(effort="low")

    print("request received")
    print(body.reasoning)

    use_browser_tool = any(
        getattr(tool, "type", None) in ("browser_search", "web_search")
        for tool in (body.tools or [])
    )
    use_code_interpreter = any(
        getattr(tool, "type", None) == "code_interpreter"
        for tool in (body.tools or [])
    )

    if use_browser_tool:
        tool_backend = os.getenv("BROWSER_BACKEND", "exa")
        if tool_backend == "youcom":
            backend = YouComBackend(source="web")
        elif tool_backend == "exa":
            backend = ExaBackend(source="web")
        else:
            raise ValueError(f"Invalid tool backend: {tool_backend}")
        browser_tool = SimpleBrowserTool(backend=backend)
    else:
        browser_tool = None

    if use_code_interpreter:
        python_tool = PythonTool()
    else:
        python_tool = None

    python_function_name_conflict = any(
        getattr(tool, "type", None) == "function"
        and getattr(tool, "name", None) == "python"
        for tool in (body.tools or [])
    )
    functions_python_as_builtin = use_code_interpreter and not (
        python_function_name_conflict
    )

    if body.previous_response_id:
        prev = responses_store.get(body.previous_response_id)
        if prev:
            prev_req, prev_resp = prev

            def _ensure_list(inp):
                if isinstance(inp, str):
                    return [
                        Item(
                            type="message",
                            role="user",
                            content=[TextContentItem(type="input_text", text=inp)],
                        )
                    ]
                return list(inp)

            merged_input = _ensure_list(prev_req.input) + list(prev_resp.output)
            merged_input.extend(_ensure_list(body.input))

            if body.instructions is None:
                body.instructions = prev_req.instructions
            body.input = merged_input

    system_message_content = SystemContent.new().with_conversation_start_date(
        datetime.datetime.now().strftime("%Y-%m-%d")
    )

    if body.reasoning is not None:
        try:

            reasoning_effort = get_reasoning_effort(body.reasoning.effort)
        except ValueError as e:
            from fastapi import HTTPException
            print(e)

            raise HTTPException(status_code=422, detail=str(e))
        system_message_content = system_message_content.with_reasoning_effort(
            reasoning_effort
        )

    if use_browser_tool:
        system_message_content = system_message_content.with_tools(
            browser_tool.tool_config
        )
    if use_code_interpreter:
        system_message_content = system_message_content.with_tools(
            python_tool.tool_config
        )

    system_message = Message.from_role_and_content(
        Role.SYSTEM, system_message_content
    )
    messages = [system_message]

    instructions = body.instructions
    assert instructions

    if instructions or body.tools:
        instructions = SYSTEM_INSTRUCTIONS_OVERRIDE
        developer_message_content = DeveloperContent.new().with_instructions(
            instructions
        )
        tools = []
        # blacklisting these that don't work well with gpt-oss due to the
        # limited effective context
        blacklisted_tools = [
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
            "view_image",
            "update_plan",
        ]
        for tool in body.tools:
            if tool.type == "function":
                if tool.name in blacklisted_tools:
                    continue
                name = tool.name
                parameters = tool.parameters
                description = tool.description or ""
                tools.append(
                    ToolDescription.new(
                        name,
                        description,
                        parameters,
                    )
                )

        if tools:
            developer_message_content = (
                developer_message_content.with_function_tools(tools)
            )

        developer_message = Message.from_role_and_content(
            Role.DEVELOPER, developer_message_content
        )

        messages.append(developer_message)

    if isinstance(body.input, str):
        user_message = Message.from_role_and_content(Role.USER, body.input)
        messages.append(user_message)
    else:
        is_last_message_function_call_output = (
            len(body.input) > 0 and body.input[-1].type == "function_call_output"
        )
        function_call_map = {}
        # Find the index of the last assistant message
        last_assistant_idx = -1
        for idx, item in enumerate(body.input):
            if item.type == "message" and item.role == Role.ASSISTANT:
                last_assistant_idx = idx

        for idx, item in enumerate(body.input):
            if item.type == "message":
                # TODO: add system prompt handling
                if isinstance(item.content, str):
                    messages.append(
                        Message.from_role_and_content(item.role, item.content)
                    )
                else:
                    for content_item in item.content:
                        messages.append(
                            Message.from_role_and_content(
                                item.role, content_item.text
                            )
                        )
                # add final channel to the last assistant message if it's from the assistant
                if item.role == Role.ASSISTANT:
                    messages[-1] = messages[-1].with_channel("final")
            elif item.type == "reasoning":
                # Only include reasoning if it is after the last assistant message and we are handling a function call at the moment
                if (
                    idx > last_assistant_idx
                    and is_last_message_function_call_output
                ):
                    for content_item in item.content:
                        messages.append(
                            Message.from_role_and_content(
                                Role.ASSISTANT, content_item.text
                            ).with_channel("analysis")
                        )
            elif item.type == "function_call":
                function_call_map[item.call_id] = item
                messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, item.arguments)
                    .with_recipient(f"functions.{item.name}")
                    .with_channel("commentary")
                )
            elif item.type == "function_call_output":
                function_call = function_call_map.get(item.call_id, None)
                if not function_call:
                    raise ValueError(f"Function call {item.call_id} not found")

                messages.append(
                    Message.from_author_and_content(
                        Author.new(Role.TOOL, f"functions.{function_call.name}"),
                        item.output,
                    )
                    .with_recipient("assistant")
                    .with_channel("commentary")
                )

    conversation = Conversation.from_messages(messages)

    initial_tokens = encoding.render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    print(encoding.decode_utf8(initial_tokens))
    response_id = f"resp_{uuid.uuid4().hex}"

    def store_callback(rid: str, req: ResponsesRequest, resp: ResponseObject):
        responses_store[rid] = (req, resp)

    event_stream = StreamResponsesEvents(
        initial_tokens,
        body,
        as_sse=body.stream,
        request=request,
        response_id=response_id,
        store_callback=store_callback,
        browser_tool=browser_tool,
        python_tool=python_tool,
        functions_python_as_builtin=functions_python_as_builtin,
    )

    if body.stream:
        return StreamingResponse(event_stream.run(), media_type="text/event-stream")
    else:
        last_event = None
        async for event in event_stream.run():
            last_event = event

        return last_event.response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')

