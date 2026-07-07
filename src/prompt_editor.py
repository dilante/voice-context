from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

try:
    from .editor_providers import ProviderError, call_api, call_ollama, call_openai_compatible
    from .utils import editor_provider_name, sanitize_text
except ImportError:
    from editor_providers import ProviderError, call_api, call_ollama, call_openai_compatible
    from utils import editor_provider_name, sanitize_text


VOICE_OP_APPEND = "append"
VOICE_OP_PATCH = "patch"
VOICE_OP_UNDO = "undo"
VOICE_OP_IGNORE = "ignore"
PROMPT_VERSION = "correction_ops_v10_direct_ops"
PROMPT_VARIANT = "direct_position_ops_no_normalizer"

SEGMENT_INLINE_PATTERN = re.compile(r"\[segment\s+(\d+)([^\]]*)\]")
META_TEST_PATTERNS = [
    re.compile(r"^(嗯|啊|呃|哦|喂)[。！!？?，, ]*$"),
    re.compile(r"(你好|哈喽|hello|听见吗|听得到吗|能听到吗|能不能听到)", re.IGNORECASE),
    re.compile(r"(测试|试一下|试试|录音测试|麦克风测试)"),
    re.compile(r"这个不错", re.IGNORECASE),
    re.compile(r"^(好的|可以|行|收到|ok|okay)[。！!？?，, ]*$", re.IGNORECASE),
    re.compile(r"(那这有什么意思|听见了吗|确认一下)"),
]
TASK_SIGNAL_PATTERNS = [
    re.compile(r"(继续|优化|修|改|处理|调整|隐藏|显示|展开|折叠|默认|界面|listen|prompt|clipboard|clip|segment|debug|状态|预览|latest prompt|latest generation)", re.IGNORECASE),
    re.compile(r"(不要|只显示|保留|切换|按\s*[vsdrqcpnl])", re.IGNORECASE),
]
PROTECTED_TERM_PATTERNS = [
    re.compile(r"\blisten\b", re.IGNORECASE),
    re.compile(r"\blatest prompt\b", re.IGNORECASE),
    re.compile(r"\blatest generation\b", re.IGNORECASE),
]
CONFLICTING_PROTECTED_TERMS = {
    "latest prompt": ["default prompt"],
    "listen": ["lesson"],
    "latest generation": ["regeneration"],
}
APPEND_PROMPT_LEAK_MARKERS = [
    "primary_language",
    "secondary_languages",
    "mixed_language_mode",
    "stt_homophone_hint",
    "current_goal",
    "latest_dynamic",
    "important_terms",
    "possible_files",
    "recent_assistant_hints",
    "term_bank_hints",
    "current_draft_tail",
    "raw_stt",
    "language_context",
    "json schema",
    "schema",
    "system",
    "assistant",
    "user",
    "prompt",
    "新的 block 文本",
    "输出",
    "任务",
    "上下文",
    "语言上下文",
]
APPEND_COMMAND_MARKERS = [
    "删除",
    "删掉",
    "去掉",
    "移除",
    "改成",
    "改为",
    "修改",
    "替换",
    "change",
    "replace",
    "remove",
    "delete",
]
APPEND_TARGET_REFERENCE_PATTERNS = [
    re.compile(r"第[一二三四五六七八九十两\d]+[句块条项行个]?"),
    re.compile(r"最后[一二三四五六七八九十两\d]*[句块条项行个]?"),
    re.compile(r"(first|second|third|last)\s+(sentence|block|item|line)?", re.IGNORECASE),
]

DEFAULT_APPEND_SYSTEM_PROMPT = """你是 listen 模式下的新 Draft block 整理器。
你只负责把当前这段 raw STT 整理成要追加到 Draft 尾部的新 Draft blocks。
只输出用户真正想表达的 Draft 内容。
绝不要输出说明文字、指令、字段名、标题、上下文标签、配置值、schema、示例或 prompt 本身。
绝不要回显 prompt 里的字段名。
不要重写已有 Draft，不要返回完整 Draft，不要解释，不要 JSON，不要 Markdown code block。
保留 raw STT 中已经成立的技术术语、UI 标签、命令名、键名、变量名和代码风格词。
不要把有效技术词替换成发音相近但无关的词。
只有 raw_stt 本身或 term_whitelist 明确支持时，才修正明显 STT 错误；如果不确定，优先保留 raw wording。
不要把普通中文短语翻译成英文，也不要把普通英文短语翻译成中文；除非用户明确要求翻译或替换语言。
只输出新的 Draft block 文本；如果需要多个 blocks，用空行分隔。"""

DEFAULT_CORRECTION_SYSTEM_PROMPT = """你是 Draft block 编辑器。
用户提供的是修改请求，不是新增内容。
只返回 JSON，绝不解释。
禁止把编辑说明写进 Draft block。"""

DEFAULT_NUMBER_NORMALIZER_SYSTEM_PROMPT = """Historical helper. Not used in the active correction path."""

DEFAULT_REPAIR_SYSTEM_PROMPT = """You repair a JSON response for a Draft editing pipeline.
Return exactly one corrected JSON object.
No markdown. No code fence. No explanation."""

DEFAULT_APPEND_USER_TEMPLATE = """你只做 raw STT 的最小清理，用于 append 新 Draft 文本。
不要执行命令，不要推断用户想编辑现有 Draft，不要根据其他上下文改写任务。
只允许：标点、断句、去口头禅、轻微语法整理、明显 ASR 错词修正。
优先做最小转写清理，不要改写成另一种说法，不要在中文字符之间插入空格。
不要删除/替换/重排已有 Draft，不要把一句话改写成别的任务。
如果 raw STT 里有 delete/remove/change/replace/删除/改成 这类词，必须按字面保留，不要替用户执行。
只输出要 append 的新文本；如需多个 blocks，用空行分隔；不要 JSON，不要解释。
primary_language={primary_language}
secondary_languages={secondary_languages}
mixed_language_mode={mixed_language_mode}
stt_homophone_hint={stt_homophone_hint}

term_whitelist={term_bank_hints}

raw_stt:
{raw_stt}"""

DEFAULT_CORRECTION_USER_TEMPLATE = """You are a Draft edit ops planner.
Return exactly one JSON object.
No markdown. No code fence. No explanation.
The only top-level keys are changed, reason, ops.
The ops are authoritative. The reason is only a short human-readable label.

You will see BLOCKS_TABLE.
The number inside [] is the visible position.
Use visible positions in every op.
Do not output block ids.
Do not invent hidden ids.
Do not use any position not shown in BLOCKS_TABLE.

Core rule:
If the user refers to a sentence, block, line, item, row, or entry in any language, map it to a visible position from BLOCKS_TABLE.
If you cannot confidently map the target to a visible position, return noop.

Accepted ops:
- replace_text: {"op":"replace_text","position":2,"old_exact":"...","new_text":"..."}
- replace_block: {"op":"replace_block","position":2,"text":"..."}
- delete_blocks: {"op":"delete_blocks","positions":[2,4]}
- swap_blocks: {"op":"swap_blocks","a_position":3,"b_position":4}
- move_block: {"op":"move_block","position":4,"to":"front"}
- move_block: {"op":"move_block","position":4,"to":"end"}
- move_block: {"op":"move_block","position":4,"to":"before","ref_position":1}
- move_block: {"op":"move_block","position":4,"to":"after","ref_position":1}
- noop: {"op":"noop"}

Use raw_user_modification_request as the correction request.
Do not invent positions.
Do not use positions that are not shown in BLOCKS_TABLE.

Replacement mapping:
- First map the target to a visible position.
- Then extract old_exact and new_text.
- old_exact is the smallest exact existing substring to replace.
- new_text is only the replacement fragment.
- Do not include target locator words in old_exact or new_text.
- Do not use the whole block as old_exact unless the user explicitly asks to rewrite or replace the entire block.
- If the user asks to change one word or phrase, use replace_text, not replace_block.
- If still unsure, return noop.

Delete:
- If the user asks to delete/remove/drop/clear a target, use delete_blocks.
- positions is always an array.

Order mapping:
- If the user asks to swap/exchange/switch two blocks, use swap_blocks.
- If the user asks to move one block to another position, use move_block.
- Do not calculate a full final order list.
- Do not output block ids.
- If unsure whether the request is swap or move, return noop.

Language rules:
- Preserve ordinary phrase language.
- Do not translate ordinary Chinese to English.
- Do not translate ordinary English to Chinese.
- Preserve technical/UI/code/model terms and whitelisted terms.

No-op rules:
- If the requested edit would make no change, return changed=false with noop.
- If the target is unclear, return changed=false with noop.
- If exact replacement cannot be safely located, return changed=false with noop.

primary_language: {primary_language}
secondary_languages: {secondary_languages}
mixed_language_mode: {mixed_language_mode}
stt_homophone_hint: {stt_homophone_hint}
term_whitelist: {term_bank_hints}

BLOCKS_TABLE:
{blocks_table_text}

raw_user_modification_request:
{raw_correction_text}"""

DEFAULT_CORRECTION_REPAIR_USER_TEMPLATE = """Repair the JSON response for this stage.
Return exactly one corrected JSON object.
No markdown. No code fence. No explanation.
Keep the same accepted ops schema.
Fix only the validation issue.
If unsafe or ambiguous, return {"changed":false,"reason":"noop","ops":[{"op":"noop"}]}.

stage:
{repair_stage}

required_contract:
{stage_contract_text}

validation_error:
{validation_error}

raw_user_modification_request:
{raw_correction_text}

BLOCKS_TABLE:
{blocks_table_text}

parsed_candidate_if_any:
{parsed_candidate_json}

previous_llm_response:
{previous_llm_response}"""


def _append_unique(items: list[str], text: str) -> None:
    normalized = text.strip()
    if normalized and normalized not in items:
        items.append(normalized)


def _voice_queue_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("voice_queue", {}) if isinstance(config, dict) else {}


def _editor_prompt_config(config: dict[str, Any]) -> dict[str, Any]:
    top_level_prompts = config.get("prompts", {}) if isinstance(config, dict) else {}
    if isinstance(top_level_prompts, dict) and top_level_prompts:
        return top_level_prompts
    editor_config = config.get("editor", {}) if isinstance(config, dict) else {}
    prompts = editor_config.get("prompts", {})
    return prompts if isinstance(prompts, dict) else {}


def _get_prompt_text(config: dict[str, Any], key: str, default: str) -> str:
    value = _editor_prompt_config(config).get(key, default)
    text = str(value or "").strip()
    return text or default


def _render_prompt_template(template: str, values: dict[str, Any]) -> str:
    rendered = str(template)
    for key in [
        "raw_stt",
        "primary_language",
        "secondary_languages",
        "mixed_language_mode",
        "stt_homophone_hint",
        "term_bank_hints",
        "blocks_table_text",
        "raw_correction_text",
        "previous_llm_response",
        "validation_error",
        "stage_contract_text",
        "repair_stage",
        "correction_text",
        "parsed_candidate_json",
    ]:
        rendered = rendered.replace("{" + key + "}", str(values.get(key, "")))
    return rendered


def _build_blocks_table_text(draft_blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for position, block in enumerate(draft_blocks, start=1):
        block_text = _normalize_block_text(str(block.get("text", "")))
        lines.append(f"[{position}] {block_text}")
    return "\n".join(lines) if lines else "(empty)"


def extract_arabic_position_numbers(text: str) -> list[int]:
    normalized = sanitize_text(text, max_chars=1200)
    numbers: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\d+", normalized):
        try:
            value = int(match.group(0))
        except Exception:
            continue
        if value > 0 and value not in seen:
            seen.add(value)
            numbers.append(value)
    return numbers


def _command_keywords(config: dict[str, Any], key: str, defaults: list[str]) -> list[str]:
    queue_config = _voice_queue_config(config)
    command_keywords = queue_config.get("command_keywords", {})
    if not isinstance(command_keywords, dict):
        return defaults
    values = command_keywords.get(key, defaults)
    if not isinstance(values, list):
        return defaults
    cleaned = [sanitize_text(item, max_chars=40).strip().lower() for item in values if str(item).strip()]
    return cleaned or defaults


def _remove_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```[^\n]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _parse_segment_blocks(stt_text: str) -> list[str]:
    text = str(stt_text or "").strip()
    if not text:
        return []
    matches = list(SEGMENT_INLINE_PATTERN.finditer(text))
    if not matches:
        return []
    segments: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = sanitize_text(text[start:end].strip(), max_chars=1200)
        if body:
            segments.append(body)
    return segments


def _normalized_stt_text(stt_text: str) -> str:
    segments = _parse_segment_blocks(stt_text)
    if segments:
        return " ".join(segments)
    return sanitize_text(stt_text)


def _strip_leading_filler(text: str) -> str:
    return re.sub(r"^(?:嗯|呃|啊|哦|那个|就是|然后|诶)[，,、 ]*", "", text.strip(), flags=re.IGNORECASE)


def _trim_command_prefix(text: str, prefixes: list[str]) -> str:
    normalized = _strip_leading_filler(text)
    lowered = normalized.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            trimmed = normalized[len(prefix):].lstrip("，,：:。 ")
            if trimmed.startswith("一下"):
                trimmed = trimmed[2:].lstrip("，,：:。 ")
            return trimmed.strip()
    return normalized


def _clean_value(value: str, *, max_chars: int = 120) -> str:
    cleaned = sanitize_text(value, max_chars=max_chars).strip().strip("`'\"“”‘’.,，。;；:： ")
    return re.sub(r"\s+", " ", cleaned)


def _is_meta_test_text(text: str) -> bool:
    normalized = sanitize_text(text, max_chars=200).strip()
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) > 24:
        return False
    return any(pattern.search(normalized) for pattern in META_TEST_PATTERNS)


def is_empty_or_punctuation_only(text: str) -> bool:
    normalized = sanitize_text(text, max_chars=1200).strip()
    if not normalized:
        return True
    for char in normalized:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if not category or category[0] not in {"P", "S", "Z", "C"}:
            return False
    return True


def _is_empty_correction_request(text: str) -> bool:
    normalized = sanitize_text(text, max_chars=200).strip()
    if is_empty_or_punctuation_only(normalized):
        return True
    without_punct = re.sub(r"[\s\.\,\!\?\;\:，。！？；：、~…\-]+", "", normalized)
    if not without_punct:
        return True
    if re.fullmatch(r"(嗯|啊|呃|哦|喂)+", without_punct, flags=re.IGNORECASE):
        return True
    return False


def _is_correction_like_segment(text: str) -> bool:
    normalized = sanitize_text(text, max_chars=300).strip().lower()
    return any(
        marker in normalized
        for marker in [
            "改成",
            "不是",
            "删除",
            "删掉",
            "去掉",
            "替换",
            "整句",
            "我说的是",
            "i mean",
            "not ",
        ]
    )


def detect_voice_command(
    stt_text: str,
    config: dict[str, Any],
    *,
    forced_mode: str = "",
) -> dict[str, Any]:
    queue_config = _voice_queue_config(config)
    undo_keywords = _command_keywords(config, "undo", ["撤销", "回退", "undo"])
    correction_prefixes = _command_keywords(config, "correction_prefixes", ["纠错", "更正", "修正"])
    max_undo_chars = int(queue_config.get("max_undo_command_chars", 12))
    prefix_scan_chars = int(queue_config.get("prefix_scan_chars", 12))

    normalized = _strip_leading_filler(_normalized_stt_text(stt_text))
    compact = re.sub(r"\s+", "", normalized)
    lowered = normalized.lower()

    if forced_mode == "correction":
        return {
            "intent": "correction",
            "text": _trim_command_prefix(normalized, correction_prefixes),
            "forced_by_key": True,
            "triggered_by_prefix": False,
        }

    if compact and len(compact) <= max_undo_chars:
        for keyword in undo_keywords:
            if keyword and keyword.lower() in lowered:
                return {"intent": "undo", "text": keyword}

    prefix_window = lowered[:prefix_scan_chars]
    if any(prefix.lower() in prefix_window for prefix in correction_prefixes):
        return {
            "intent": "correction",
            "text": _trim_command_prefix(normalized, correction_prefixes),
            "forced_by_key": False,
            "triggered_by_prefix": True,
        }

    if _is_meta_test_text(normalized):
        return {"intent": "ignore", "text": normalized, "reason": "meta_test"}

    return {"intent": "append", "text": normalized}


def _language_context_settings(config: dict[str, Any]) -> dict[str, Any]:
    queue_config = _voice_queue_config(config)
    language_context = queue_config.get("language_context", {})
    if not isinstance(language_context, dict):
        language_context = {}
    secondary = language_context.get("secondary_languages", ["en"])
    if isinstance(secondary, str):
        secondary = [secondary]
    elif not isinstance(secondary, list):
        secondary = ["en"]
    cleaned_secondary = [sanitize_text(item, max_chars=20).strip() for item in secondary if str(item).strip()]
    mixed_language_mode = language_context.get("mixed_language_mode")
    if mixed_language_mode is None:
        mixed_language_mode = bool(cleaned_secondary)
    return {
        "primary_language": sanitize_text(language_context.get("primary_language", "zh"), max_chars=20).strip() or "zh",
        "secondary_languages": cleaned_secondary,
        "mixed_language_mode": bool(mixed_language_mode),
        "stt_homophone_hint": bool(language_context.get("stt_homophone_hint", True)),
    }


def _recent_assistant_messages(context: dict[str, Any]) -> list[str]:
    if not isinstance(context, dict):
        return []
    messages = context.get("recent_messages", [])
    if not isinstance(messages, list):
        return []
    hints: list[str] = []
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"assistant", "ai", "agent", "codex"}:
            continue
        hint = sanitize_text(item.get("summary") or item.get("content"), max_chars=300)
        if hint:
            _append_unique(hints, hint)
        if len(hints) >= 3:
            break
    return hints


def _term_bank_lines(context: dict[str, Any], *, max_categories: int = 5, max_items_each: int = 4) -> list[str]:
    term_bank = context.get("term_bank", {}) if isinstance(context, dict) else {}
    if not isinstance(term_bank, dict):
        return []
    lines: list[str] = []
    for field in ["domain_terms", "files", "functions", "classes", "variables", "commands", "ui_terms"]:
        values = term_bank.get(field, [])
        if not isinstance(values, list):
            continue
        cleaned = [sanitize_text(item, max_chars=80).strip() for item in values if str(item).strip()][:max_items_each]
        if not cleaned:
            continue
        lines.append(f"{field}: " + ", ".join(cleaned))
        if len(lines) >= max_categories:
            break
    return lines


def _term_bank_whitelist(context: dict[str, Any], *, max_items: int = 10) -> list[str]:
    term_bank = context.get("term_bank", {}) if isinstance(context, dict) else {}
    if not isinstance(term_bank, dict):
        return []
    whitelist: list[str] = []
    for field in ["domain_terms", "files", "functions", "classes", "variables", "commands", "ui_terms"]:
        values = term_bank.get(field, [])
        if not isinstance(values, list):
            continue
        for value in values:
            cleaned = _clean_value(value, max_chars=80)
            if cleaned:
                _append_unique(whitelist, cleaned)
            if len(whitelist) >= max_items:
                return whitelist
    return whitelist


def _blocks_preview(blocks: list[dict[str, Any]], max_items: int = 4) -> list[str]:
    return [f"[{block.get('id', 0)}] {sanitize_text(block.get('text', ''), max_chars=80)}" for block in blocks[:max_items]]


def _split_into_blocks(text: str) -> list[str]:
    normalized = sanitize_text(text, max_chars=1000).strip()
    if not normalized:
        return []
    parts = re.split(r"[。！？!?]\s*", normalized)
    blocks: list[str] = []
    for part in parts:
        cleaned = sanitize_text(part, max_chars=240).strip(" ，,;；")
        if not cleaned:
            continue
        if is_empty_or_punctuation_only(cleaned):
            continue
        if _is_meta_test_text(cleaned):
            continue
        _append_unique(blocks, cleaned + "。")
    return blocks


def _copy_draft_blocks(draft_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": int(block.get("id", 0)), "text": str(block.get("text", ""))} for block in draft_blocks]


def _reindex_blocks(draft_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reindexed: list[dict[str, Any]] = []
    for index, block in enumerate(draft_blocks, start=1):
        text = sanitize_text(block.get("text", ""), max_chars=400).strip()
        if not text:
            continue
        reindexed.append({"id": index, "text": text})
    return reindexed


def _join_draft_blocks(draft_blocks: list[dict[str, Any]]) -> str:
    return " ".join(
        sanitize_text(str(block.get("text", "")).strip(), max_chars=400).strip()
        for block in draft_blocks
        if str(block.get("text", "")).strip()
    ).strip()


def _next_block_id(draft_blocks: list[dict[str, Any]]) -> int:
    return max((int(block.get("id", 0)) for block in draft_blocks), default=0) + 1


def _normalize_block_text(text: str) -> str:
    cleaned = sanitize_text(text, max_chars=400).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _same_block_layout(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    left_pairs = [(int(block.get("id", 0)), _normalize_block_text(str(block.get("text", "")))) for block in left]
    right_pairs = [(int(block.get("id", 0)), _normalize_block_text(str(block.get("text", "")))) for block in right]
    return left_pairs == right_pairs


def _make_diff_item(kind: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"kind": kind}
    payload.update(kwargs)
    return payload


def _compute_block_diff(old_blocks: list[dict[str, Any]], new_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old_map = {int(block.get("id", 0)): str(block.get("text", "")) for block in old_blocks}
    new_map = {int(block.get("id", 0)): str(block.get("text", "")) for block in new_blocks}
    diff_items: list[dict[str, Any]] = []
    old_order = [int(block.get("id", 0)) for block in old_blocks]
    new_order = [int(block.get("id", 0)) for block in new_blocks]
    for old_id in old_order:
        if old_id not in new_map:
            diff_items.append(_make_diff_item("delete", block_id=old_id, old=old_map[old_id]))
    for block in new_blocks:
        block_id = int(block.get("id", 0))
        text = str(block.get("text", ""))
        if block_id not in old_map:
            diff_items.append(_make_diff_item("add", block_id=block_id, text=text))
        elif _normalize_block_text(old_map[block_id]) != _normalize_block_text(text):
            diff_items.append(_make_diff_item("replace", block_id=block_id, old=old_map[block_id], new=text, text=text))
    if not diff_items and old_order != new_order:
        for index, block_id in enumerate(new_order, start=1):
            diff_items.append(_make_diff_item("replace", block_id=block_id, old="reordered", new=f"slot {index}", text=new_map[block_id]))
    return diff_items


def _fallback_append_operation(text: str) -> list[str]:
    clean_text = _strip_leading_filler(text)
    blocks = _split_into_blocks(clean_text)
    if blocks:
        return blocks
    fallback_text = sanitize_text(clean_text, max_chars=240).strip()
    if not fallback_text or is_empty_or_punctuation_only(fallback_text):
        return []
    return [fallback_text]


def _extract_command_markers(text: str) -> list[str]:
    normalized = sanitize_text(text, max_chars=1200).lower()
    hits: list[str] = []
    for marker in APPEND_COMMAND_MARKERS:
        if marker.lower() in normalized:
            hits.append(marker.lower())
    return hits


def _extract_target_references(text: str) -> list[str]:
    normalized = sanitize_text(text, max_chars=1200)
    hits: list[str] = []
    for pattern in APPEND_TARGET_REFERENCE_PATTERNS:
        for match in pattern.finditer(normalized):
            reference = _clean_value(match.group(0), max_chars=60).lower()
            if reference:
                _append_unique(hits, reference)
    return hits


def _contains_draft_content_leak(raw_text: str, append_blocks: list[str], draft_blocks: list[dict[str, Any]]) -> bool:
    raw_lower = sanitize_text(raw_text, max_chars=1200).lower()
    output_lower = "\n".join(append_blocks).lower()
    for block in draft_blocks:
        draft_text = _normalize_block_text(str(block.get("text", "")))
        if not draft_text:
            continue
        draft_norm = draft_text.lower()
        if draft_norm in output_lower and draft_norm not in raw_lower:
            return True
        draft_terms = _extract_english_like_terms(draft_text)
        for term in draft_terms:
            lowered = term.lower()
            if lowered in output_lower and lowered not in raw_lower:
                return True
    return False


def _semantic_drift_detected(raw_text: str, append_blocks: list[str]) -> bool:
    raw_norm = _normalize_block_text(raw_text)
    output_norm = _normalize_block_text(" ".join(append_blocks))
    if not raw_norm or not output_norm:
        return True
    raw_letters = [term.lower() for term in _extract_english_like_terms(raw_norm)]
    output_letters = {term.lower() for term in _extract_english_like_terms(output_norm)}
    if raw_letters:
        missing = [term for term in raw_letters if term not in output_letters]
        if len(missing) >= max(1, len(raw_letters) // 2):
            return True
    raw_len = len(raw_norm)
    output_len = len(output_norm)
    if raw_len >= 12 and output_len < max(4, int(raw_len * 0.5)):
        return True
    ratio = SequenceMatcher(None, raw_norm.lower(), output_norm.lower()).ratio()
    return ratio < 0.45


def _extract_protected_terms(segments: list[str]) -> list[str]:
    protected_terms: list[str] = []
    for segment in segments:
        normalized = sanitize_text(segment, max_chars=500)
        for pattern in PROTECTED_TERM_PATTERNS:
            for match in pattern.finditer(normalized):
                term = _clean_value(match.group(0), max_chars=80)
                if term:
                    _append_unique(protected_terms, term)
    return protected_terms


def _extract_english_like_terms(text: str) -> list[str]:
    terms: list[str] = []
    normalized = sanitize_text(text, max_chars=1200)
    for match in re.finditer(r"\b(?:[A-Za-z][A-Za-z0-9_-]{1,}|[A-Z])\b", normalized):
        term = _clean_value(match.group(0), max_chars=80)
        if term:
            _append_unique(terms, term)
    return terms


def _protected_term_violation(prompt: str, protected_terms: list[str]) -> str:
    lowered_prompt = prompt.lower()
    for term in protected_terms:
        lowered_term = term.lower()
        for conflict in CONFLICTING_PROTECTED_TERMS.get(lowered_term, []):
            if conflict in lowered_prompt:
                return f"输出错误使用了 {conflict}，应优先使用 {term}。"
    return ""


def _contains_prompt_leak(text: str) -> bool:
    lowered = sanitize_text(text, max_chars=800).lower()
    for marker in APPEND_PROMPT_LEAK_MARKERS:
        if marker.lower() in lowered:
            return True
    return False


def _append_output_hygiene_issue(
    raw_text: str,
    append_blocks: list[str],
    protected_terms: list[str],
    draft_blocks: list[dict[str, Any]],
) -> str:
    if not append_blocks:
        return "append_model_output_invalid"
    for block in append_blocks:
        if _contains_prompt_leak(block):
            return "append_prompt_leak_rejected"
    raw_lower = sanitize_text(raw_text, max_chars=1200).lower()
    raw_terms = _extract_english_like_terms(raw_text)
    protected_pool = protected_terms[:] or []
    for term in raw_terms:
        _append_unique(protected_pool, term)

    combined_output = "\n".join(append_blocks)
    output_terms = _extract_english_like_terms(combined_output)
    output_lower_terms = {term.lower() for term in output_terms}

    for term in protected_pool:
        lowered = term.lower()
        if lowered in raw_lower and lowered not in output_lower_terms:
            return "append_protected_term_changed"

    output_lower = combined_output.lower()
    for marker in _extract_command_markers(raw_text):
        if marker not in output_lower:
            return "append_command_word_dropped"

    for reference in _extract_target_references(raw_text):
        if reference not in output_lower:
            return "append_target_reference_dropped"

    raw_lower_terms = {term.lower() for term in raw_terms}
    for output_term in output_terms:
        lowered = output_term.lower()
        if lowered in raw_lower_terms:
            continue
        for raw_term in raw_terms:
            ratio = SequenceMatcher(None, lowered, raw_term.lower()).ratio()
            if 0.72 <= ratio < 1.0:
                return "append_suspicious_term"

    protected_violation = _protected_term_violation(combined_output, protected_terms)
    if protected_violation:
        return "append_protected_term_changed"
    if _contains_draft_content_leak(raw_text, append_blocks, draft_blocks):
        return "append_draft_content_leak"
    if _semantic_drift_detected(raw_text, append_blocks):
        return "append_semantic_drift"
    return ""

def _normalize_append_model_output(raw_content: str) -> list[str]:
    cleaned = _remove_code_fences(raw_content).strip()
    if not cleaned:
        return []
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return []
    blocks: list[str] = []
    chunks = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    if not chunks:
        chunks = [cleaned]
    for chunk in chunks:
        normalized_chunk = sanitize_text(chunk, max_chars=800).strip()
        if not normalized_chunk:
            continue
        if is_empty_or_punctuation_only(normalized_chunk):
            continue
        extracted = _split_into_blocks(normalized_chunk)
        if extracted:
            for block in extracted:
                _append_unique(blocks, block)
        else:
            text_block = normalized_chunk if normalized_chunk.endswith("。") else normalized_chunk + "。"
            if is_empty_or_punctuation_only(text_block):
                continue
            _append_unique(blocks, text_block)
    return blocks


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _remove_code_fences(text).strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _end = decoder.raw_decode(cleaned[match.start() :])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _operation_result(
    *,
    op: str,
    draft_blocks: list[dict[str, Any]],
    diff_items: list[dict[str, Any]],
    prompt: str = "",
    fallback: bool = False,
    fallback_reason: str = "",
    ignored: bool = False,
    reason: str = "",
    raw_text: str = "",
    editor_elapsed_ms: int = 0,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_blocks = _reindex_blocks(draft_blocks)
    final_prompt = prompt or _join_draft_blocks(normalized_blocks)
    return {
        "op": op,
        "draft_blocks": _copy_draft_blocks(normalized_blocks),
        "diff_items": diff_items,
        "prompt": final_prompt,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "ignored": ignored,
        "reason": reason,
        "raw_text": sanitize_text(raw_text, max_chars=300),
        "editor_elapsed_ms": editor_elapsed_ms,
        "debug": debug or {},
    }


def _provider_debug_info(
    provider_result: dict[str, Any] | None,
    *,
    mode: str,
) -> dict[str, Any]:
    result = provider_result or {}
    return {
        "mode": sanitize_text(mode, max_chars=40),
        "model": sanitize_text(result.get("model", ""), max_chars=120),
        "resolved_model": sanitize_text(result.get("resolved_model", result.get("model", "")), max_chars=160),
        "base_url": sanitize_text(result.get("base_url", ""), max_chars=240),
        "base_url_host": sanitize_text(result.get("base_url_host", ""), max_chars=120),
        "provider_name": sanitize_text(result.get("provider_name", ""), max_chars=80),
        "status_code": int(result.get("status_code", 0) or 0),
        "usage": result.get("usage", {}) if isinstance(result.get("usage", {}), dict) else {},
        "api_key_masked": sanitize_text(result.get("api_key_masked", ""), max_chars=80),
        "api_extra_body_keys": result.get("api_extra_body_keys", []) if isinstance(result.get("api_extra_body_keys", []), list) else [],
        "provider_elapsed_ms": int(result.get("provider_elapsed_ms", 0) or 0),
        "no_think_enabled": bool(result.get("no_think_enabled", result.get("qwen3_no_think_enabled", False))),
        "qwen3_no_think_enabled": bool(result.get("qwen3_no_think_enabled", result.get("no_think_enabled", False))),
        "no_think_injected": bool(result.get("no_think_injected", False)),
    }


def _build_llm_append_prompt(
    stt_text: str,
    context: dict[str, Any],
    config: dict[str, Any],
    draft_blocks: list[dict[str, Any]],
) -> str:
    language = _language_context_settings(config)
    whitelist = _term_bank_whitelist(context)
    template = _get_prompt_text(config, "append_user_template", DEFAULT_APPEND_USER_TEMPLATE)
    return _render_prompt_template(
        template,
        {
            "raw_stt": sanitize_text(stt_text, max_chars=600),
            "primary_language": language["primary_language"],
            "secondary_languages": ", ".join(language["secondary_languages"]) if language["secondary_languages"] else "none",
            "mixed_language_mode": str(language["mixed_language_mode"]),
            "stt_homophone_hint": str(language["stt_homophone_hint"]),
            "term_bank_hints": ", ".join(whitelist) if whitelist else "none",
            "draft_blocks_json": "",
            "correction_text": "",
        },
    )


def _build_llm_correction_prompt(
    correction_text: str,
    draft_blocks: list[dict[str, Any]],
    context: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    language = _language_context_settings(config)
    whitelist = _term_bank_whitelist(context)
    blocks_table_text = _build_blocks_table_text(draft_blocks)
    template = _get_prompt_text(config, "correction_user_template", DEFAULT_CORRECTION_USER_TEMPLATE)
    return (
        _render_prompt_template(
            template,
            {
                "raw_stt": "",
                "primary_language": language["primary_language"],
                "secondary_languages": ", ".join(language["secondary_languages"]) if language["secondary_languages"] else "none",
                "mixed_language_mode": str(language["mixed_language_mode"]),
                "stt_homophone_hint": str(language["stt_homophone_hint"]),
                "term_bank_hints": ", ".join(whitelist) if whitelist else "none",
                "blocks_table_text": blocks_table_text,
                "raw_correction_text": sanitize_text(correction_text, max_chars=500),
            },
        ),
        language,
    )


def _build_correction_repair_prompt(
    *,
    raw_correction_text: str,
    draft_blocks: list[dict[str, Any]],
    previous_llm_response: str,
    validation_error: str,
    repair_stage: str,
    stage_contract_text: str,
    parsed_candidate_json: str,
    config: dict[str, Any],
) -> str:
    template = _get_prompt_text(config, "correction_repair_user_template", DEFAULT_CORRECTION_REPAIR_USER_TEMPLATE)
    return _render_prompt_template(
        template,
        {
            "raw_correction_text": sanitize_text(raw_correction_text, max_chars=500),
            "blocks_table_text": _build_blocks_table_text(draft_blocks),
            "previous_llm_response": sanitize_text(previous_llm_response, max_chars=4000),
            "validation_error": sanitize_text(validation_error, max_chars=240),
            "repair_stage": sanitize_text(repair_stage, max_chars=80),
            "stage_contract_text": sanitize_text(stage_contract_text, max_chars=3000),
            "parsed_candidate_json": sanitize_text(parsed_candidate_json, max_chars=4000),
        },
    )


def _call_editor_provider(
    *,
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    missing_mode_error: str,
) -> tuple[dict[str, Any] | None, str, str]:
    editor_config = config.get("editor", {})
    mode = editor_provider_name(config)
    temperature = float(editor_config.get("temperature", 0))
    timeout_sec = float(editor_config.get("timeout_sec", 2.0))
    if mode == "ollama":
        return (
            call_ollama(
                config=config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout_sec=timeout_sec,
            ),
            mode,
            "",
        )
    if mode not in {"rules", "openai_compatible"}:
        return (
            call_api(
                config=config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout_sec=timeout_sec,
            ),
            mode,
            "",
        )
    if mode == "openai_compatible":
        return (
            call_openai_compatible(
                config=config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout_sec=timeout_sec,
            ),
            mode,
            "",
        )
    return None, mode, missing_mode_error


def _call_append_provider(prompt_text: str, config: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    return _call_editor_provider(
        config=config,
        system_prompt=_get_prompt_text(config, "append_system", DEFAULT_APPEND_SYSTEM_PROMPT),
        user_prompt=prompt_text,
        missing_mode_error="append_requires_model",
    )


def _call_correction_provider(prompt_text: str, config: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    return _call_editor_provider(
        config=config,
        system_prompt=_get_prompt_text(config, "correction_system", DEFAULT_CORRECTION_SYSTEM_PROMPT),
        user_prompt=prompt_text,
        missing_mode_error="llm_correction_requires_model",
    )


def _call_number_normalizer_provider(prompt_text: str, config: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    return _call_editor_provider(
        config=config,
        system_prompt=_get_prompt_text(config, "number_normalizer_system", DEFAULT_NUMBER_NORMALIZER_SYSTEM_PROMPT),
        user_prompt=prompt_text,
        missing_mode_error="llm_number_normalizer_requires_model",
    )


def _call_correction_repair_provider(prompt_text: str, config: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    return _call_editor_provider(
        config=config,
        system_prompt=_get_prompt_text(config, "correction_repair_system", DEFAULT_REPAIR_SYSTEM_PROMPT),
        user_prompt=prompt_text,
        missing_mode_error="llm_correction_repair_requires_model",
    )


def build_append_operation(
    stt_text: str,
    context: dict[str, Any],
    config: dict[str, Any],
    draft_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_text = _strip_leading_filler(_normalized_stt_text(stt_text))
    fallback_blocks = _fallback_append_operation(clean_text)
    protected_terms = _extract_protected_terms([clean_text])
    editor_mode = editor_provider_name(config)
    append_system_prompt = _get_prompt_text(config, "append_system", DEFAULT_APPEND_SYSTEM_PROMPT)
    if is_empty_or_punctuation_only(clean_text):
        return _operation_result(
            op=VOICE_OP_IGNORE,
            draft_blocks=_copy_draft_blocks(draft_blocks),
            diff_items=[],
            ignored=True,
            reason="append_empty_or_punctuation_only",
            raw_text=clean_text,
            debug={
                "mode": "append",
                "prompt_version": PROMPT_VERSION,
                "prompt_variant": PROMPT_VARIANT,
                "raw_text": sanitize_text(clean_text, max_chars=240),
                "system_prompt": "",
                "user_prompt": "",
                "llm_raw_response": "",
                "parsed_response": {},
                "parsed_ops": [],
                "provider": {"mode": "rules", "model": "", "base_url": ""},
                "qwen3_no_think_enabled": False,
                "no_think_injected": False,
                "editor_model": "",
                "provider_elapsed_ms": 0,
                "validation_error": "append_empty_or_punctuation_only",
            },
        )
    if editor_mode == "rules":
        if not fallback_blocks:
            return _operation_result(
                op=VOICE_OP_IGNORE,
                draft_blocks=_copy_draft_blocks(draft_blocks),
                diff_items=[],
                ignored=True,
                reason="append_output_empty_or_punctuation_only",
                raw_text=clean_text,
                debug={
                    "mode": "append",
                    "prompt_version": PROMPT_VERSION,
                    "prompt_variant": PROMPT_VARIANT,
                    "raw_text": sanitize_text(clean_text, max_chars=240),
                    "system_prompt": "",
                    "user_prompt": "",
                    "llm_raw_response": "",
                    "parsed_response": {},
                    "parsed_ops": [],
                    "provider": {"mode": "rules", "model": "", "base_url": ""},
                    "qwen3_no_think_enabled": False,
                    "no_think_injected": False,
                    "editor_model": "",
                    "provider_elapsed_ms": 0,
                    "validation_error": "append_output_empty_or_punctuation_only",
                },
            )
        new_draft = _copy_draft_blocks(draft_blocks)
        next_id = _next_block_id(new_draft)
        diff_items: list[dict[str, Any]] = []
        for block_text in fallback_blocks:
            block = {"id": next_id, "text": block_text}
            next_id += 1
            new_draft.append(block)
            diff_items.append(_make_diff_item("add", block_id=block["id"], text=block_text))
        return _operation_result(
            op=VOICE_OP_APPEND,
            draft_blocks=new_draft,
            diff_items=diff_items,
            raw_text=clean_text,
            debug={
                "mode": "append",
                "prompt_version": PROMPT_VERSION,
                "prompt_variant": PROMPT_VARIANT,
                "raw_text": sanitize_text(clean_text, max_chars=240),
                "system_prompt": "",
                "user_prompt": "",
                "llm_raw_response": "",
                "parsed_response": {},
                "parsed_ops": [],
                "provider": {"mode": "rules", "model": "", "base_url": ""},
                "qwen3_no_think_enabled": False,
                "no_think_injected": False,
                "editor_model": "",
                "provider_elapsed_ms": 0,
            },
        )

    started = time.perf_counter()
    raw_content = ""
    append_blocks: list[str] = []
    provider_result: dict[str, Any] | None = None
    prompt_text = ""
    output_fallback_reason = ""
    try:
        prompt_text = _build_llm_append_prompt(clean_text, context, config, draft_blocks)
        provider_result, _mode, mode_error = _call_append_provider(prompt_text, config)
        if mode_error:
            raise ProviderError(mode_error)
        assert provider_result is not None
        raw_content = str(provider_result.get("raw_content", ""))
        append_blocks = _normalize_append_model_output(raw_content)
        if not append_blocks:
            if fallback_blocks:
                append_blocks = fallback_blocks
                output_fallback_reason = "append_output_empty_or_punctuation_only"
            else:
                return _operation_result(
                    op=VOICE_OP_IGNORE,
                    draft_blocks=_copy_draft_blocks(draft_blocks),
                    diff_items=[],
                    ignored=True,
                    reason="append_output_empty_or_punctuation_only",
                    raw_text=clean_text,
                    editor_elapsed_ms=int((time.perf_counter() - started) * 1000),
                    debug={
                        "mode": "append",
                        "prompt_version": PROMPT_VERSION,
                        "prompt_variant": PROMPT_VARIANT,
                        "raw_text": sanitize_text(clean_text, max_chars=240),
                        "system_prompt": sanitize_text(append_system_prompt, max_chars=6000),
                        "user_prompt": sanitize_text(prompt_text, max_chars=6000),
                        "llm_raw_response": sanitize_text(raw_content, max_chars=6000),
                        "raw_content_preview": sanitize_text(raw_content, max_chars=500),
                        "cleaned_content_preview": "",
                        "protected_terms": protected_terms,
                        "parsed_response": {},
                        "parsed_ops": [],
                        "provider": _provider_debug_info(provider_result, mode=editor_mode),
                        "qwen3_no_think_enabled": bool(provider_result.get("qwen3_no_think_enabled", False)),
                        "no_think_injected": bool(provider_result.get("no_think_injected", False)),
                        "editor_model": sanitize_text(provider_result.get("model", ""), max_chars=120),
                        "provider_elapsed_ms": int(provider_result.get("provider_elapsed_ms", 0) or 0),
                        "validation_error": "append_output_empty_or_punctuation_only",
                    },
                )
        hygiene_issue = _append_output_hygiene_issue(clean_text, append_blocks, protected_terms, draft_blocks)
        if hygiene_issue:
            raise ProviderError(hygiene_issue)
        new_draft = _copy_draft_blocks(draft_blocks)
        next_id = _next_block_id(new_draft)
        diff_items: list[dict[str, Any]] = []
        for block_text in append_blocks:
            block = {"id": next_id, "text": block_text}
            next_id += 1
            new_draft.append(block)
            diff_items.append(_make_diff_item("add", block_id=block["id"], text=block_text))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return _operation_result(
            op=VOICE_OP_APPEND,
            draft_blocks=new_draft,
            diff_items=diff_items,
            raw_text=clean_text,
            fallback=bool(output_fallback_reason),
            fallback_reason=output_fallback_reason,
            editor_elapsed_ms=elapsed_ms,
            debug={
                "mode": "append",
                "prompt_version": PROMPT_VERSION,
                "prompt_variant": PROMPT_VARIANT,
                "raw_text": sanitize_text(clean_text, max_chars=240),
                "system_prompt": sanitize_text(append_system_prompt, max_chars=6000),
                "user_prompt": sanitize_text(prompt_text, max_chars=6000),
                "llm_raw_response": sanitize_text(raw_content, max_chars=6000),
                "raw_content_preview": sanitize_text(raw_content, max_chars=500),
                "cleaned_content_preview": sanitize_text("\n\n".join(append_blocks), max_chars=500),
                "protected_terms": protected_terms,
                "parsed_response": {},
                "parsed_ops": [],
                "provider": _provider_debug_info(provider_result, mode=editor_mode),
                "qwen3_no_think_enabled": bool(provider_result.get("qwen3_no_think_enabled", False)),
                "no_think_injected": bool(provider_result.get("no_think_injected", False)),
                "editor_model": sanitize_text(provider_result.get("model", ""), max_chars=120),
                "provider_elapsed_ms": int(provider_result.get("provider_elapsed_ms", 0) or 0),
            },
        )
    except ProviderError as exc:
        provider_debug = _provider_debug_info(provider_result, mode=editor_mode)
        new_draft = _copy_draft_blocks(draft_blocks)
        next_id = _next_block_id(new_draft)
        diff_items: list[dict[str, Any]] = []
        for block_text in fallback_blocks:
            block = {"id": next_id, "text": block_text}
            next_id += 1
            new_draft.append(block)
            diff_items.append(_make_diff_item("add", block_id=block["id"], text=block_text))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return _operation_result(
            op=VOICE_OP_APPEND,
            draft_blocks=new_draft,
            diff_items=diff_items,
            raw_text=clean_text,
            fallback=True,
            fallback_reason=sanitize_text(str(exc), max_chars=160),
            editor_elapsed_ms=elapsed_ms,
            debug={
                "mode": "append",
                "prompt_version": PROMPT_VERSION,
                "prompt_variant": PROMPT_VARIANT,
                "raw_text": sanitize_text(clean_text, max_chars=240),
                "system_prompt": sanitize_text(append_system_prompt, max_chars=6000),
                "user_prompt": sanitize_text(prompt_text, max_chars=6000),
                "llm_raw_response": sanitize_text(raw_content, max_chars=6000),
                "raw_content_preview": sanitize_text(raw_content, max_chars=500),
                "cleaned_content_preview": sanitize_text("\n\n".join(append_blocks), max_chars=500),
                "protected_terms": protected_terms,
                "parsed_response": {},
                "parsed_ops": [],
                "provider": provider_debug,
                "qwen3_no_think_enabled": bool(provider_debug.get("qwen3_no_think_enabled", False)),
                "no_think_injected": bool(provider_debug.get("no_think_injected", False)),
                "editor_model": sanitize_text(provider_debug.get("model", ""), max_chars=120),
                "provider_elapsed_ms": int(provider_debug.get("provider_elapsed_ms", 0) or 0),
            },
        )


def _collapse_spelled_letters(text: str) -> str:
    normalized = _normalize_block_text(text)
    if not normalized:
        return ""
    pattern = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z](?:\s+[A-Za-z]){1,})(?![A-Za-z0-9_])")
    return pattern.sub(lambda match: re.sub(r"\s+", "", match.group(1)), normalized)


def _text_multiset(blocks: list[dict[str, Any]]) -> Counter[str]:
    return Counter(_normalize_block_text(str(block.get("text", ""))) for block in blocks)


def _safe_replace_text(
    *,
    block: dict[str, Any],
    op_item: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    current_text = _normalize_block_text(str(block.get("text", "")))
    old_exact = _normalize_block_text(str(op_item.get("old_exact", op_item.get("old", ""))))
    new_text = _normalize_block_text(str(op_item.get("new_text", op_item.get("new", ""))))
    if not old_exact or not new_text:
        return current_text, {}, "correction_invalid_op"

    normalization_used = False
    normalized_old = old_exact
    normalized_new = new_text
    replacement_target = old_exact

    direct_occurrences = current_text.count(replacement_target) if replacement_target else 0
    if direct_occurrences > 1:
        return current_text, {
            "normalization_used": False,
            "normalized_old": normalized_old,
            "normalized_new": normalized_new,
        }, "correction_old_text_ambiguous"
    if replacement_target and replacement_target == current_text:
        return current_text, {
            "normalization_used": False,
            "normalized_old": normalized_old,
            "normalized_new": normalized_new,
        }, "correction_replace_text_uses_whole_block"

    if direct_occurrences <= 0:
        collapsed_old = _collapse_spelled_letters(old_exact)
        collapsed_new = _collapse_spelled_letters(new_text)
        if collapsed_old != old_exact:
            normalization_used = True
            normalized_old = collapsed_old
        if collapsed_new != new_text:
            normalization_used = True
            normalized_new = collapsed_new
        if normalized_old == normalized_new:
            return current_text, {
                "normalization_used": normalization_used,
                "normalized_old": normalized_old,
                "normalized_new": normalized_new,
            }, "noop_correction"
        occurrences = current_text.count(normalized_old) if normalized_old else 0
        if occurrences <= 0:
            return current_text, {
                "normalization_used": normalization_used,
                "normalized_old": normalized_old,
                "normalized_new": normalized_new,
            }, "correction_old_text_not_found"
        if occurrences > 1:
            return current_text, {
                "normalization_used": normalization_used,
                "normalized_old": normalized_old,
                "normalized_new": normalized_new,
            }, "correction_old_text_ambiguous"
        if normalized_old == current_text:
            return current_text, {
                "normalization_used": normalization_used,
                "normalized_old": normalized_old,
                "normalized_new": normalized_new,
            }, "correction_replace_text_uses_whole_block"
        replacement_target = normalized_old
        new_text = normalized_new

    if replacement_target == new_text:
        return current_text, {
            "normalization_used": normalization_used,
            "normalized_old": normalized_old,
            "normalized_new": normalized_new,
        }, "noop_correction"

    replaced_text = current_text.replace(replacement_target, new_text, 1)
    if replaced_text == current_text:
        return current_text, {
            "normalization_used": normalization_used,
            "normalized_old": normalized_old,
            "normalized_new": normalized_new,
        }, "noop_correction"

    debug_update = {}
    if normalization_used:
        debug_update = {
            "normalization_used": True,
            "normalized_old": normalized_old,
            "normalized_new": normalized_new,
        }
    return replaced_text, debug_update, ""


def _swap_blocks_in_list(
    blocks: list[dict[str, Any]],
    a: int,
    b: int,
) -> tuple[list[dict[str, Any]], str]:
    if a == b:
        return _copy_draft_blocks(blocks), "correction_invalid_op"
    working = _copy_draft_blocks(blocks)
    index_map = {int(block.get("id", 0)): idx for idx, block in enumerate(working)}
    if a not in index_map or b not in index_map:
        return _copy_draft_blocks(blocks), "correction_target_not_found"
    left_index = index_map[a]
    right_index = index_map[b]
    working[left_index], working[right_index] = working[right_index], working[left_index]
    return working, ""


def _move_block_in_list(
    blocks: list[dict[str, Any]],
    block_id: int,
    to: str,
    ref_id: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if to not in {"front", "end", "before", "after"}:
        return _copy_draft_blocks(blocks), "correction_invalid_op"
    working = _copy_draft_blocks(blocks)
    index_map = {int(block.get("id", 0)): idx for idx, block in enumerate(working)}
    if block_id not in index_map:
        return _copy_draft_blocks(blocks), "correction_target_not_found"
    if to in {"before", "after"}:
        if ref_id is None:
            return _copy_draft_blocks(blocks), "correction_invalid_op"
        if ref_id not in index_map:
            return _copy_draft_blocks(blocks), "correction_target_not_found"
        if ref_id == block_id:
            return _copy_draft_blocks(blocks), "correction_invalid_op"
    moving = working.pop(index_map[block_id])
    if to == "front":
        target_index = 0
    elif to == "end":
        target_index = len(working)
    else:
        ref_index = next(idx for idx, block in enumerate(working) if int(block.get("id", 0)) == ref_id)
        target_index = ref_index if to == "before" else ref_index + 1
    working.insert(target_index, moving)
    return working, ""


def _block_id_at_position(blocks: list[dict[str, Any]], position: int) -> int | None:
    if position < 1 or position > len(blocks):
        return None
    try:
        return int(blocks[position - 1].get("id", 0))
    except Exception:
        return None


def _validate_number_normalization_response(
    response_data: dict[str, Any],
    raw_text: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    normalized_request = sanitize_text(response_data.get("normalized_request", ""), max_chars=1200).strip()
    if is_empty_or_punctuation_only(raw_text):
        return {
            "normalized_request": "",
        }, "", {
            "normalized_request": "",
            "extracted_position_numbers": [],
        }
    if set(response_data.keys()) != {"normalized_request"}:
        return response_data, "number_normalization_invalid_contract", {}
    if not isinstance(response_data.get("normalized_request", ""), str):
        return response_data, "number_normalization_invalid_contract", {}
    if not normalized_request:
        return response_data, "number_normalization_invalid_contract", {}
    extracted_numbers = extract_arabic_position_numbers(normalized_request)
    normalized = {
        "normalized_request": normalized_request,
    }
    return normalized, "", {
        "normalized_request": normalized_request,
        "extracted_position_numbers": extracted_numbers,
    }


def _compile_order_preview(
    proposed_order: list[int],
    block_count: int,
) -> tuple[dict[str, Any] | None, list[int], dict[str, Any], str]:
    expected_order = list(range(1, block_count + 1))
    if sorted(proposed_order) != expected_order:
        return None, [], {"proposed_order": proposed_order, "original_order": expected_order}, "correction_invalid_op"
    if proposed_order == expected_order:
        return {"op": "noop"}, [], {"proposed_order": proposed_order, "original_order": expected_order}, ""

    differing_positions = [idx + 1 for idx, (left, right) in enumerate(zip(expected_order, proposed_order)) if left != right]
    order_preview_diff = {
        "proposed_order": proposed_order,
        "original_order": expected_order,
        "differing_slots": differing_positions,
    }

    if len(differing_positions) == 2:
        first_idx = differing_positions[0] - 1
        second_idx = differing_positions[1] - 1
        if (
            proposed_order[first_idx] == expected_order[second_idx]
            and proposed_order[second_idx] == expected_order[first_idx]
        ):
            a_position = expected_order[first_idx]
            b_position = expected_order[second_idx]
            return (
                {"op": "swap_blocks", "a_position": a_position, "b_position": b_position},
                sorted({a_position, b_position}),
                order_preview_diff,
                "",
            )

    for moved_position in expected_order:
        original_without = [item for item in expected_order if item != moved_position]
        proposed_without = [item for item in proposed_order if item != moved_position]
        if original_without != proposed_without:
            continue
        new_index = proposed_order.index(moved_position)
        old_index = expected_order.index(moved_position)
        if new_index == old_index:
            continue
        if new_index == 0:
            return (
                {"op": "move_block", "position": moved_position, "to": "front"},
                [moved_position],
                order_preview_diff,
                "",
            )
        if new_index == len(proposed_order) - 1:
            return (
                {"op": "move_block", "position": moved_position, "to": "end"},
                [moved_position],
                order_preview_diff,
                "",
            )
        ref_position = proposed_order[new_index - 1]
        return (
            {"op": "move_block", "position": moved_position, "to": "after", "ref_position": ref_position},
            sorted({moved_position, ref_position}),
            order_preview_diff,
            "",
        )

    return None, [], order_preview_diff, "correction_order_preview_not_clean"


def _normalize_position_ops(
    response_data: dict[str, Any],
    old_blocks: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if set(response_data.keys()) != {"changed", "reason", "ops"}:
        return response_data, "correction_invalid_op", {"llm_position_ops": [], "normalized_internal_ops": [], "used_positions": []}
    if not isinstance(response_data.get("changed"), bool):
        return response_data, "correction_invalid_op", {"llm_position_ops": [], "normalized_internal_ops": [], "used_positions": []}
    if not isinstance(response_data.get("reason"), str):
        return response_data, "correction_invalid_op", {"llm_position_ops": [], "normalized_internal_ops": [], "used_positions": []}
    ops = response_data.get("ops")
    if not isinstance(ops, list):
        return response_data, "correction_invalid_op", {"llm_position_ops": [], "normalized_internal_ops": [], "used_positions": []}
    normalized_ops: list[dict[str, Any]] = []
    used_positions: set[int] = set()
    compiled_internal_ops: list[dict[str, Any]] = []
    block_count = len(old_blocks)
    for op_item in ops:
        if not isinstance(op_item, dict):
            return response_data, "correction_invalid_op", {
                "llm_position_ops": ops,
                "normalized_internal_ops": normalized_ops,
                "used_positions": sorted(used_positions),
            }
        op_name = str(op_item.get("op", "")).strip().lower()
        if op_name == "noop":
            if set(op_item.keys()) != {"op"}:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            normalized_ops.append({"op": "noop"})
            compiled_internal_ops.append({"op": "noop"})
            continue
        if op_name == "replace_text":
            if set(op_item.keys()) != {"op", "position", "old_exact", "new_text"}:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            try:
                position = int(op_item.get("position"))
            except Exception:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            if position < 1 or position > block_count:
                return response_data, "correction_target_not_found", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(set(used_positions).union({position})),
                }
            block_id = _block_id_at_position(old_blocks, position)
            if block_id is None or block_id <= 0:
                return response_data, "correction_target_not_found", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(set(used_positions).union({position})),
                }
            normalized_ops.append(
                {
                    "op": "replace_text",
                    "id": block_id,
                    "old_exact": op_item.get("old_exact", ""),
                    "new_text": op_item.get("new_text", ""),
                }
            )
            compiled_internal_ops.append(normalized_ops[-1])
            used_positions.add(position)
            continue
        if op_name == "replace_block":
            if set(op_item.keys()) != {"op", "position", "text"}:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            try:
                position = int(op_item.get("position"))
            except Exception:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            if position < 1 or position > block_count:
                return response_data, "correction_target_not_found", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(set(used_positions).union({position})),
                }
            block_id = _block_id_at_position(old_blocks, position)
            if block_id is None or block_id <= 0:
                return response_data, "correction_target_not_found", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(set(used_positions).union({position})),
                }
            normalized_ops.append({"op": "replace_block", "id": block_id, "text": op_item.get("text", "")})
            compiled_internal_ops.append(normalized_ops[-1])
            used_positions.add(position)
            continue
        if op_name == "delete_blocks":
            if set(op_item.keys()) != {"op", "positions"}:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            positions = op_item.get("positions")
            if not isinstance(positions, list) or not positions:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            try:
                normalized_positions = [int(item) for item in positions]
            except Exception:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            if len(set(normalized_positions)) != len(normalized_positions):
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            delete_ids: list[int] = []
            for position in normalized_positions:
                if position < 1 or position > block_count:
                    return response_data, "correction_target_not_found", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(set(used_positions).union(normalized_positions)),
                    }
                block_id = _block_id_at_position(old_blocks, position)
                if block_id is None or block_id <= 0:
                    return response_data, "correction_target_not_found", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(set(used_positions).union(normalized_positions)),
                    }
                delete_ids.append(block_id)
            normalized_ops.append({"op": "delete_blocks", "ids": delete_ids})
            compiled_internal_ops.append(normalized_ops[-1])
            used_positions.update(normalized_positions)
            continue
        if op_name == "swap_blocks":
            if set(op_item.keys()) != {"op", "a_position", "b_position"}:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            try:
                a_position = int(op_item.get("a_position"))
                b_position = int(op_item.get("b_position"))
            except Exception:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            if a_position < 1 or a_position > block_count or b_position < 1 or b_position > block_count:
                return response_data, "correction_target_not_found", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(set(used_positions).union({a_position, b_position})),
                }
            a_id = _block_id_at_position(old_blocks, a_position)
            b_id = _block_id_at_position(old_blocks, b_position)
            if a_id is None or b_id is None or a_id <= 0 or b_id <= 0:
                return response_data, "correction_target_not_found", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(set(used_positions).union({a_position, b_position})),
                }
            if a_position == b_position:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(set(used_positions).union({a_position, b_position})),
                }
            normalized_ops.append({"op": "swap_blocks", "a": a_id, "b": b_id})
            compiled_internal_ops.append(normalized_ops[-1])
            used_positions.update({a_position, b_position})
            continue
        if op_name == "move_block":
            move_keys = set(op_item.keys())
            try:
                position = int(op_item.get("position"))
            except Exception:
                return response_data, "correction_invalid_op", {
                    "llm_position_ops": ops,
                    "normalized_internal_ops": normalized_ops,
                    "used_positions": sorted(used_positions),
                }
            to = str(op_item.get("to", "")).strip().lower()
            if to in {"front", "end"}:
                if move_keys != {"op", "position", "to"}:
                    return response_data, "correction_invalid_op", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(used_positions),
                    }
                if position < 1 or position > block_count:
                    return response_data, "correction_target_not_found", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(set(used_positions).union({position})),
                    }
                block_id = _block_id_at_position(old_blocks, position)
                if block_id is None or block_id <= 0:
                    return response_data, "correction_target_not_found", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(set(used_positions).union({position})),
                    }
                normalized_ops.append({"op": "move_block", "id": block_id, "to": to})
                compiled_internal_ops.append(normalized_ops[-1])
                used_positions.add(position)
                continue
            if to in {"before", "after"}:
                if move_keys != {"op", "position", "to", "ref_position"}:
                    return response_data, "correction_invalid_op", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(used_positions),
                    }
                try:
                    ref_position = int(op_item.get("ref_position"))
                except Exception:
                    return response_data, "correction_invalid_op", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(used_positions),
                    }
                if position == ref_position:
                    return response_data, "correction_invalid_op", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(set(used_positions).union({position, ref_position})),
                    }
                if position < 1 or position > block_count or ref_position < 1 or ref_position > block_count:
                    return response_data, "correction_target_not_found", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(set(used_positions).union({position, ref_position})),
                    }
                block_id = _block_id_at_position(old_blocks, position)
                ref_id = _block_id_at_position(old_blocks, ref_position)
                if block_id is None or ref_id is None or block_id <= 0 or ref_id <= 0:
                    return response_data, "correction_target_not_found", {
                        "llm_position_ops": ops,
                        "normalized_internal_ops": normalized_ops,
                        "used_positions": sorted(set(used_positions).union({position, ref_position})),
                    }
                normalized_ops.append({"op": "move_block", "id": block_id, "to": to, "ref_id": ref_id})
                compiled_internal_ops.append(normalized_ops[-1])
                used_positions.update({position, ref_position})
                continue
            return response_data, "correction_invalid_op", {
                "llm_position_ops": ops,
                "normalized_internal_ops": normalized_ops,
                "used_positions": sorted(used_positions),
            }
        return response_data, "correction_invalid_op", {
            "llm_position_ops": ops,
            "normalized_internal_ops": normalized_ops,
            "used_positions": sorted(used_positions),
        }
    normalized_response = dict(response_data)
    normalized_response["ops"] = normalized_ops
    return normalized_response, "", {
        "llm_position_ops": ops,
        "normalized_internal_ops": normalized_ops,
        "compiled_internal_ops": compiled_internal_ops,
        "used_positions": sorted(used_positions),
    }


def _apply_correction_ops(
    response_data: dict[str, Any],
    old_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    if "ops" not in response_data:
        return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", {}
    ops = response_data.get("ops")
    if not isinstance(ops, list):
        return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", {}
    if not ops:
        return _copy_draft_blocks(old_blocks), [], "noop_correction", {}

    working = _copy_draft_blocks(old_blocks)
    index_map = {int(block.get("id", 0)): block for block in working}
    changed = False
    debug_updates: dict[str, Any] = {}
    for op_item in ops:
        if not isinstance(op_item, dict):
            return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
        op_name = str(op_item.get("op", "")).strip().lower()
        if op_name not in {"replace_text", "replace_block", "delete_blocks", "swap_blocks", "move_block", "noop"}:
            return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
        if op_name == "noop":
            return _copy_draft_blocks(old_blocks), [], "noop_correction", debug_updates

        if op_name == "swap_blocks":
            if set(op_item.keys()) != {"op", "a", "b"}:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            try:
                a = int(op_item.get("a"))
                b = int(op_item.get("b"))
            except Exception:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            swapped, swap_error = _swap_blocks_in_list(working, a, b)
            if swap_error:
                return _copy_draft_blocks(old_blocks), [], swap_error, debug_updates
            if _same_block_layout(_reindex_blocks(working), _reindex_blocks(swapped)):
                return _copy_draft_blocks(old_blocks), [], "noop_correction", debug_updates
            working = swapped
            index_map = {int(block.get("id", 0)): block for block in working}
            changed = True
            continue

        if op_name == "move_block":
            move_keys = set(op_item.keys())
            try:
                block_id = int(op_item.get("id"))
            except Exception:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            to = str(op_item.get("to", "")).strip().lower()
            if to in {"front", "end"}:
                if move_keys != {"op", "id", "to"}:
                    return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            elif to in {"before", "after"}:
                if move_keys != {"op", "id", "to", "ref_id"}:
                    return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            else:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            ref_id = op_item.get("ref_id")
            if ref_id is not None:
                try:
                    ref_id = int(ref_id)
                except Exception:
                    return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            moved, move_error = _move_block_in_list(working, block_id, to, ref_id)
            if move_error:
                return _copy_draft_blocks(old_blocks), [], move_error, debug_updates
            if _same_block_layout(_reindex_blocks(working), _reindex_blocks(moved)):
                return _copy_draft_blocks(old_blocks), [], "noop_correction", debug_updates
            working = moved
            index_map = {int(block.get("id", 0)): block for block in working}
            changed = True
            continue

        if op_name == "delete_blocks":
            if set(op_item.keys()) != {"op", "ids"}:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            ids = op_item.get("ids")
            if not isinstance(ids, list) or not ids:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            try:
                delete_ids = [int(item) for item in ids]
            except Exception:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            if len(set(delete_ids)) != len(delete_ids):
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            if any(block_id not in index_map for block_id in delete_ids):
                return _copy_draft_blocks(old_blocks), [], "correction_target_not_found", debug_updates
            working = [item for item in working if int(item.get("id", 0)) not in set(delete_ids)]
            index_map = {int(block.get("id", 0)): block for block in working}
            changed = True
            continue

        if op_name == "replace_text":
            if set(op_item.keys()) != {"op", "id", "old_exact", "new_text"}:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
        elif op_name == "replace_block":
            if set(op_item.keys()) != {"op", "id", "text"}:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates

        block_id = op_item.get("id")
        try:
            block_id = int(block_id)
        except Exception:
            return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
        if block_id <= 0:
            return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
        if block_id not in index_map:
            return _copy_draft_blocks(old_blocks), [], "correction_target_not_found", debug_updates

        block = index_map[block_id]
        current_text = _normalize_block_text(str(block.get("text", "")))
        if op_name == "replace_block":
            new_text = _normalize_block_text(str(op_item.get("text", "")))
            if not new_text:
                return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
            if new_text == current_text:
                return _copy_draft_blocks(old_blocks), [], "noop_correction", debug_updates
            block["text"] = new_text
            changed = True
            continue

        replaced_text, replace_debug, replace_error = _safe_replace_text(block=block, op_item=op_item)
        debug_updates.update(replace_debug)
        if replace_error:
            return _copy_draft_blocks(old_blocks), [], replace_error, debug_updates
        block["text"] = replaced_text
        changed = True

    original_reindexed = _reindex_blocks(old_blocks)
    reindexed = _reindex_blocks(working)
    if any(str(item.get("op", "")).strip().lower() in {"swap_blocks", "move_block"} for item in ops if isinstance(item, dict)):
        if len(reindexed) != len(original_reindexed):
            return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
        if _text_multiset(reindexed) != _text_multiset(original_reindexed):
            return _copy_draft_blocks(old_blocks), [], "correction_invalid_op", debug_updates
    if not changed or _same_block_layout(original_reindexed, reindexed):
        return _copy_draft_blocks(old_blocks), [], "noop_correction", debug_updates
    diff_items = _compute_block_diff(original_reindexed, reindexed)
    return reindexed, diff_items, "", debug_updates


REPAIRABLE_CORRECTION_ERRORS = {
    "invalid_json",
    "correction_invalid_op",
    "correction_target_not_found",
    "correction_old_text_not_found",
    "correction_old_text_ambiguous",
    "correction_replace_text_uses_whole_block",
}


def _correction_contract_text() -> str:
    return (
        '{"changed":true,"reason":"...","ops":[{"op":"replace_text","position":2,"old_exact":"...","new_text":"..."}]} '
        'or replace_block/delete_blocks/swap_blocks/move_block/noop using visible positions only. '
        "Use only positions shown in BLOCKS_TABLE. Do not invent positions."
    )


def _record_correction_attempt(
    attempts: list[dict[str, Any]],
    *,
    attempt: int,
    stage: str,
    llm_raw_response: str,
    parsed_response: Any,
    validation_error: str,
    used_positions: list[int] | None = None,
    provider_elapsed_ms: int = 0,
    repair_reason: str = "",
) -> None:
    entry: dict[str, Any] = {
        "attempt": attempt,
        "stage": stage,
        "llm_raw_response": sanitize_text(llm_raw_response, max_chars=6000),
        "parsed_response": parsed_response if isinstance(parsed_response, (dict, list)) else {},
        "validation_error": sanitize_text(validation_error, max_chars=200),
    }
    if used_positions is not None:
        entry["used_positions"] = used_positions
    if provider_elapsed_ms:
        entry["per_attempt_elapsed_ms"] = int(provider_elapsed_ms)
    if repair_reason:
        entry["repair_reason"] = sanitize_text(repair_reason, max_chars=200)
    attempts.append(entry)


def build_llm_correction_operation(
    stt_text: str,
    context: dict[str, Any],
    config: dict[str, Any],
    draft_blocks: list[dict[str, Any]],
    *,
    forced_by_key: bool = False,
    triggered_by_prefix: bool = False,
) -> dict[str, Any]:
    working = _copy_draft_blocks(draft_blocks)
    clean_text = _trim_command_prefix(_normalized_stt_text(stt_text), _command_keywords(config, "correction_prefixes", ["纠错", "更正", "修正"]))
    correction_system_prompt = _get_prompt_text(config, "correction_system", DEFAULT_CORRECTION_SYSTEM_PROMPT)
    max_repair_attempts = max(int(config.get("editor", {}).get("correction_max_repair_attempts", 3) or 3), 0)
    if is_empty_or_punctuation_only(clean_text) or _is_empty_correction_request(clean_text):
        return _operation_result(
            op=VOICE_OP_IGNORE,
            draft_blocks=working,
            diff_items=[],
            ignored=True,
            reason="correction_empty_or_punctuation_only",
            raw_text=clean_text,
            debug={
                "correction_raw_text": sanitize_text(clean_text, max_chars=240),
                "prompt_version": PROMPT_VERSION,
                "prompt_variant": PROMPT_VARIANT,
                "correction_forced_by_key": forced_by_key,
                "correction_triggered_by_prefix": triggered_by_prefix,
                "system_prompt": "",
                "user_prompt": "",
                "llm_raw_response": "",
                "parsed_response": {},
                "parsed_ops": [],
                "llm_position_ops": [],
                "normalized_internal_ops": [],
                "compiled_internal_ops": [],
                "correction_attempts": [],
                "raw_user_modification_request": sanitize_text(clean_text, max_chars=240),
                "used_positions": [],
                "correction_planner_raw_response": "",
                "correction_planner_parsed_response": {},
                "final_validation_error": "correction_empty_or_punctuation_only",
                "provider": {"mode": "rules", "model": "", "base_url": ""},
                "qwen3_no_think_enabled": False,
                "no_think_injected": False,
                "editor_model": "",
                "provider_elapsed_ms": 0,
                "validation_error": "correction_empty_or_punctuation_only",
                "llm_correction_raw_response": "",
                "llm_correction_changed": False,
                "llm_correction_reason": "correction_empty_or_punctuation_only",
                "old_blocks_preview": _blocks_preview(working, max_items=12),
                "new_blocks_preview": [],
                "diff_items": [],
            },
        )
    language = _language_context_settings(config)
    correction_attempts: list[dict[str, Any]] = []
    debug_payload = {
        "correction_raw_text": sanitize_text(clean_text, max_chars=240),
        "prompt_version": PROMPT_VERSION,
        "prompt_variant": PROMPT_VARIANT,
        "correction_forced_by_key": forced_by_key,
        "correction_triggered_by_prefix": triggered_by_prefix,
        "language_primary": language["primary_language"],
        "language_secondary": language["secondary_languages"],
        "system_prompt": sanitize_text(correction_system_prompt, max_chars=6000),
        "user_prompt": "",
        "llm_raw_response": "",
        "parsed_response": {},
        "parsed_ops": [],
        "llm_position_ops": [],
        "normalized_internal_ops": [],
        "compiled_internal_ops": [],
        "provider": {"mode": "", "model": "", "base_url": ""},
        "qwen3_no_think_enabled": False,
        "no_think_injected": False,
        "editor_model": "",
        "provider_elapsed_ms": 0,
        "llm_correction_prompt_preview": "",
        "llm_correction_raw_response": "",
        "llm_correction_changed": False,
        "llm_correction_reason": "",
        "old_blocks_preview": _blocks_preview(working, max_items=12),
        "new_blocks_preview": [],
        "diff_items": [],
        "validation_error": "",
        "final_validation_error": "",
        "raw_user_modification_request": sanitize_text(clean_text, max_chars=240),
        "used_positions": [],
        "correction_planner_raw_response": "",
        "correction_planner_parsed_response": {},
        "correction_attempts": correction_attempts,
    }
    try:
        correction_prompt, _language = _build_llm_correction_prompt(clean_text, working, context, config)
        debug_payload["llm_correction_prompt_preview"] = sanitize_text(correction_prompt, max_chars=500)
        debug_payload["user_prompt"] = sanitize_text(correction_prompt, max_chars=6000)
        debug_payload["system_prompt"] = sanitize_text(correction_system_prompt, max_chars=6000)

        current_prompt = correction_prompt
        last_reason = ""

        for attempt in range(1, max_repair_attempts + 2):
            if attempt == 1:
                result, _mode, mode_error = _call_correction_provider(current_prompt, config)
            else:
                result, _mode, mode_error = _call_correction_repair_provider(current_prompt, config)
            debug_payload["provider"] = _provider_debug_info(result, mode=_mode)
            debug_payload["qwen3_no_think_enabled"] = bool(result.get("qwen3_no_think_enabled", False)) if result else False
            debug_payload["no_think_injected"] = bool(result.get("no_think_injected", False)) if result else False
            debug_payload["editor_model"] = sanitize_text(result.get("model", ""), max_chars=120) if result else ""
            debug_payload["provider_elapsed_ms"] = int(result.get("provider_elapsed_ms", 0) or 0) if result else 0
            if mode_error:
                debug_payload["validation_error"] = mode_error
                debug_payload["final_validation_error"] = mode_error
                return _operation_result(
                    op=VOICE_OP_IGNORE,
                    draft_blocks=working,
                    diff_items=[],
                    ignored=True,
                    reason=mode_error,
                    raw_text=clean_text,
                    debug=debug_payload,
                )
            assert result is not None
            raw_content = str(result.get("content", "") or result.get("raw_content", "")).strip()
            debug_payload["llm_raw_response"] = sanitize_text(raw_content, max_chars=6000)
            debug_payload["llm_correction_raw_response"] = sanitize_text(raw_content, max_chars=500)
            debug_payload["correction_planner_raw_response"] = sanitize_text(raw_content, max_chars=6000)
            response_data = _extract_json_object(raw_content)
            if not isinstance(response_data, dict):
                validation_error = "invalid_json"
                debug_payload["correction_planner_parsed_response"] = {}
                _record_correction_attempt(
                    correction_attempts,
                    attempt=attempt,
                    stage="correction_planning" if attempt == 1 else "repair",
                    llm_raw_response=raw_content,
                    parsed_response={},
                    validation_error=validation_error,
                    provider_elapsed_ms=int(result.get("provider_elapsed_ms", 0) or 0),
                    repair_reason=last_reason if attempt > 1 else "",
                )
                if attempt <= max_repair_attempts:
                    last_reason = validation_error
                    current_prompt = _build_correction_repair_prompt(
                        raw_correction_text=clean_text,
                        draft_blocks=working,
                        previous_llm_response=raw_content,
                        validation_error=validation_error,
                        repair_stage="correction_planning",
                        stage_contract_text=_correction_contract_text(),
                        parsed_candidate_json="{}",
                        config=config,
                    )
                    continue
                debug_payload["validation_error"] = validation_error
                debug_payload["final_validation_error"] = validation_error
                return _operation_result(
                    op=VOICE_OP_IGNORE,
                    draft_blocks=working,
                    diff_items=[],
                    ignored=True,
                    reason="validation_error",
                    raw_text=clean_text,
                    debug=debug_payload,
                )

            debug_payload["parsed_response"] = response_data
            debug_payload["correction_planner_parsed_response"] = response_data
            parsed_ops = response_data.get("ops", [])
            debug_payload["parsed_ops"] = parsed_ops if isinstance(parsed_ops, list) else []
            changed = bool(response_data.get("changed", False))
            reason = sanitize_text(str(response_data.get("reason", "")), max_chars=160).strip() or ("changed" if changed else "unchanged")
            debug_payload["llm_correction_changed"] = changed
            debug_payload["llm_correction_reason"] = reason

            normalized_response, normalize_error, normalize_debug = _normalize_position_ops(
                response_data,
                working,
            )
            debug_payload.update(normalize_debug)
            validation_error = normalize_error
            validated_blocks: list[dict[str, Any]] = working
            diff_items: list[dict[str, Any]] = []
            apply_debug: dict[str, Any] = {}
            if not validation_error:
                validated_blocks, diff_items, validation_error, apply_debug = _apply_correction_ops(
                    normalized_response,
                    working,
                )
                debug_payload.update(apply_debug)

            _record_correction_attempt(
                correction_attempts,
                attempt=attempt,
                stage="correction_planning" if attempt == 1 else "repair",
                llm_raw_response=raw_content,
                parsed_response=response_data,
                validation_error=validation_error,
                used_positions=list(normalize_debug.get("used_positions", [])),
                provider_elapsed_ms=int(result.get("provider_elapsed_ms", 0) or 0),
                repair_reason=last_reason if attempt > 1 else "",
            )

            if validation_error:
                if validation_error in REPAIRABLE_CORRECTION_ERRORS and attempt <= max_repair_attempts:
                    last_reason = validation_error
                    current_prompt = _build_correction_repair_prompt(
                        raw_correction_text=clean_text,
                        draft_blocks=working,
                        previous_llm_response=raw_content,
                        validation_error=validation_error,
                        repair_stage="correction_planning",
                        stage_contract_text=_correction_contract_text(),
                        parsed_candidate_json=json.dumps(response_data, ensure_ascii=False),
                        config=config,
                    )
                    continue
                debug_payload["validation_error"] = validation_error
                debug_payload["final_validation_error"] = validation_error
                return _operation_result(
                    op=VOICE_OP_IGNORE,
                    draft_blocks=working,
                    diff_items=[],
                    ignored=True,
                    reason=validation_error,
                    raw_text=clean_text,
                    debug=debug_payload,
                )

            debug_payload["new_blocks_preview"] = _blocks_preview(validated_blocks, max_items=12)
            if not changed:
                return _operation_result(
                    op=VOICE_OP_IGNORE,
                    draft_blocks=working,
                    diff_items=[],
                    ignored=True,
                    reason=reason or "unchanged",
                    raw_text=clean_text,
                    debug=debug_payload,
                )
            if _same_block_layout(working, validated_blocks):
                debug_payload["llm_correction_changed"] = False
                debug_payload["llm_correction_reason"] = "noop_correction"
                return _operation_result(
                    op=VOICE_OP_IGNORE,
                    draft_blocks=working,
                    diff_items=[],
                    ignored=True,
                    reason="noop_correction",
                    raw_text=clean_text,
                    debug=debug_payload,
                )
            debug_payload["diff_items"] = diff_items
            debug_payload["final_validation_error"] = ""
            return _operation_result(
                op=VOICE_OP_PATCH,
                draft_blocks=validated_blocks,
                diff_items=diff_items,
                raw_text=clean_text,
                debug=debug_payload,
            )

        debug_payload["validation_error"] = last_reason or "correction_repair_exhausted"
        debug_payload["final_validation_error"] = debug_payload["validation_error"]
        return _operation_result(
            op=VOICE_OP_IGNORE,
            draft_blocks=working,
            diff_items=[],
            ignored=True,
            reason=debug_payload["validation_error"],
            raw_text=clean_text,
            debug=debug_payload,
        )
    except ProviderError as exc:
        debug_payload["validation_error"] = sanitize_text(str(exc), max_chars=160)
        debug_payload["final_validation_error"] = debug_payload["validation_error"]
        debug_payload["llm_correction_reason"] = sanitize_text(str(exc), max_chars=160)
        return _operation_result(
            op=VOICE_OP_IGNORE,
            draft_blocks=working,
            diff_items=[],
            ignored=True,
            reason="provider_error",
            raw_text=clean_text,
            debug=debug_payload,
        )


def plan_voice_operation(
    stt_text: str,
    context: dict[str, Any],
    config: dict[str, Any],
    draft_blocks: list[dict[str, Any]],
    *,
    forced_mode: str = "",
) -> dict[str, Any]:
    command = detect_voice_command(stt_text, config, forced_mode=forced_mode)
    intent = str(command.get("intent", "append"))
    command_text = str(command.get("text", "")).strip()
    if intent == "ignore":
        return _operation_result(
            op=VOICE_OP_IGNORE,
            draft_blocks=draft_blocks,
            diff_items=[],
            ignored=True,
            reason=str(command.get("reason", "meta_test")),
            raw_text=command_text,
        )
    if intent == "undo":
        return _operation_result(
            op=VOICE_OP_UNDO,
            draft_blocks=draft_blocks,
            diff_items=[],
            raw_text=command_text,
            debug={"mode": "undo", "raw_text": sanitize_text(command_text, max_chars=240)},
        )
    if intent == "correction":
        return build_llm_correction_operation(
            command_text or stt_text,
            context,
            config,
            draft_blocks,
            forced_by_key=bool(command.get("forced_by_key", False)),
            triggered_by_prefix=bool(command.get("triggered_by_prefix", False)),
        )
    return build_append_operation(command_text or stt_text, context, config, draft_blocks)
