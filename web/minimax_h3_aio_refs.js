import { app } from "../../scripts/app.js";

const SPEECH_WIDGETS = [
    "speech_text",
    "expressive_tts_mode",
    "dialogue_script",
    "speaking_speed_hint",
    "voice_type",
    "custom_voice_description",
    "voice_timbre",
    "speaking_style",
    "emotion",
    "emotion_intensity",
    "speech_speed",
    "speech_pitch",
    "voice_reference_mode",
    "cloning_instructions",
    "expressive_default_style",
    "expressive_default_emotion",
    "expressive_default_intensity",
    "expressive_default_speed",
    "expressive_default_pitch",
];

function findInputIndex(node, name) {
    return (node.inputs || []).findIndex((i) => i.name === name);
}

function insertAfter(node, afterName, name, type) {
    node.addInput(name, type);
    moveInputAfter(node, name, afterName);
}

function moveInputAfter(node, name, afterName) {
    const from = findInputIndex(node, name);
    const after = findInputIndex(node, afterName);
    if (from < 0 || after < 0 || from === after + 1) return;
    const [inp] = node.inputs.splice(from, 1);
    const adjustedAfter = findInputIndex(node, afterName);
    node.inputs.splice(adjustedAfter + 1, 0, inp);
}

function getGraphLink(node, linkId) {
    if (linkId == null || !node.graph?.links) return null;
    return node.graph.links[linkId] || null;
}

function removeInputPreservingLink(node, name) {
    const idx = findInputIndex(node, name);
    if (idx < 0) return;
    const input = node.inputs[idx];
    const link = getGraphLink(node, input.link);
    const saved = {
        name: input.name,
        type: input.type,
        label: input.label,
        originalIndex: input._mmOriginalIndex ?? idx,
        link: link ? { origin_id: link.origin_id, origin_slot: link.origin_slot } : null,
    };
    node._mmRemovedVoiceInputs ||= {};
    node._mmRemovedVoiceInputs[name] = saved;
    if (input.link != null) node.disconnectInput(idx);
    node.removeInput(idx);
}

function restoreInput(node, name, type, label = null) {
    let idx = findInputIndex(node, name);
    if (idx < 0) {
        const saved = node._mmRemovedVoiceInputs?.[name];
        node.addInput(name, saved?.type || type);
        idx = node.inputs.length - 1;
        const target = Math.max(0, Math.min(saved?.originalIndex ?? idx, node.inputs.length - 1));
        if (idx !== target) {
            const [slot] = node.inputs.splice(idx, 1);
            node.inputs.splice(target, 0, slot);
        }
        if (saved?.link && node.graph) {
            const origin = node.graph.getNodeById?.(saved.link.origin_id);
            if (origin?.connect) {
                try { origin.connect(saved.link.origin_slot, node, findInputIndex(node, name)); } catch (_) {}
            }
        }
        if (node._mmRemovedVoiceInputs) delete node._mmRemovedVoiceInputs[name];
        idx = findInputIndex(node, name);
    }
    if (idx >= 0) {
        const input = node.inputs[idx];
        input.type = type;
        if (label) input.label = label;
        else delete input.label;
    }
}

function setInputSet(node, visibleNames) {
    const all = [
        { name: "ref_image", type: "IMAGE" },
        { name: "ref_audio", type: "AUDIO" },
        { name: "ref_audio_1", type: "AUDIO", label: "speaker_1" },
        { name: "ref_audio_2", type: "AUDIO", label: "speaker_2" },
    ];
    for (const spec of all) {
        if (visibleNames.has(spec.name)) restoreInput(node, spec.name, spec.type, spec.label);
        else removeInputPreservingLink(node, spec.name);
    }
    // Keep the visible speaker sockets adjacent and ordered Speaker 1 -> Speaker 2.
    if (visibleNames.has("ref_audio_1")) moveInputAfter(node, "ref_audio_1", "audio_vae");
    if (visibleNames.has("ref_audio_2")) moveInputAfter(node, "ref_audio_2", "ref_audio_1");
    if (visibleNames.has("ref_audio") && !visibleNames.has("ref_audio_1")) moveInputAfter(node, "ref_audio", "audio_vae");
    if (visibleNames.has("ref_image") && !visibleNames.has("ref_audio_1")) moveInputAfter(node, "ref_image", "audio_vae");
}

function setWidgetVisible(node, name, visible) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) return;
    w.hidden = !visible;
    if (w.element) w.element.style.display = visible ? "" : "none";
}

function resizeNode(node) {
    try {
        const size = node.computeSize();
        if (node.size) node.size[1] = size[1];
        else node.setSize([size[0], size[1]]);
    } catch (_) {}
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}


const TAG_LIBRARY = {
    "Emotions": ["[happy]", "[sad]", "[angry]", "[surprised]", "[curious]", "[excited]", "[fearful]", "[calm]", "[romantic]", "[confident]"],
    "Delivery": ["[whispering]", "[shouting]", "[softly]", "[slowly]", "[tenderly]", "[sarcastically]", "[seriously]", "[hesitantly]"],
    "Reactions": ["[laughing]", "[chuckling]", "[crying]", "[sighing]", "[gasping]", "[groaning]", "[clearing throat]"],
    "Pauses": ["[pause=300ms]", "[pause=500ms]", "[pause=700ms]", "[pause=1s]", "[pause=1.5s]", "[pause=2s]"],
};

function getTextWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function getTextElement(widget) {
    if (!widget) return null;
    if (widget.inputEl && typeof widget.inputEl.setSelectionRange === "function") return widget.inputEl;
    if (widget.element && typeof widget.element.setSelectionRange === "function") return widget.element;
    const el = widget.element?.querySelector?.("textarea, input");
    if (el) return el;
    return null;
}

function insertTagAtCursor(node, widgetName, tag, savedSelection) {
    const widget = getTextWidget(node, widgetName);
    const el = getTextElement(widget);
    if (!widget || !el) {
        // Fallback for widget implementations without an exposed textarea element.
        const oldValue = String(widget?.value ?? "");
        const next = oldValue + (oldValue && !/\s$/.test(oldValue) ? " " : "") + tag;
        if (widget) {
            widget.value = next;
            widget.callback?.(next);
            node.setDirtyCanvas?.(true, true);
        }
        return;
    }

    const start = Number.isInteger(savedSelection?.start) ? savedSelection.start :
        (Number.isInteger(el.selectionStart) ? el.selectionStart : el.value.length);
    const end = Number.isInteger(savedSelection?.end) ? savedSelection.end :
        (Number.isInteger(el.selectionEnd) ? el.selectionEnd : start);
    const before = el.value.slice(0, start);
    const after = el.value.slice(end);
    const leftSpace = before && !/\s$/.test(before) ? " " : "";
    const rightSpace = after && !/^\s/.test(after) ? " " : "";
    const insertion = leftSpace + tag + rightSpace;
    const nextValue = before + insertion + after;

    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    if (setter) setter.call(el, nextValue);
    else el.value = nextValue;
    const cursor = before.length + insertion.length;
    el.setSelectionRange(cursor, cursor);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    widget.value = nextValue;
    widget.callback?.(nextValue);
    el.focus();
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function openTagPicker(node, widgetName, buttonWidget) {
    const target = getTextWidget(node, widgetName);
    const targetEl = getTextElement(target);
    const saved = {
        start: targetEl && Number.isInteger(targetEl.selectionStart) ? targetEl.selectionStart : String(target?.value ?? "").length,
        end: targetEl && Number.isInteger(targetEl.selectionEnd) ? targetEl.selectionEnd : String(target?.value ?? "").length,
    };

    document.querySelectorAll(".mm-voice-tag-picker").forEach((el) => el.remove());
    const popup = document.createElement("div");
    popup.className = "mm-voice-tag-picker";
    Object.assign(popup.style, {
        position: "fixed", zIndex: "100000", width: "290px", maxHeight: "420px",
        overflow: "auto", padding: "12px", borderRadius: "10px",
        background: "var(--comfy-menu-bg, #242424)", color: "var(--input-text, #eee)",
        border: "1px solid var(--border-color, #555)", boxShadow: "0 8px 28px #0009",
        font: "13px sans-serif",
    });

    const title = document.createElement("div");
    title.textContent = "Insert expressive tag";
    title.style.cssText = "font-weight:700;margin-bottom:9px";
    popup.appendChild(title);

    const tabs = document.createElement("div");
    tabs.style.cssText = "display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px";
    popup.appendChild(tabs);
    const list = document.createElement("div");
    list.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px";
    popup.appendChild(list);

    const close = () => popup.remove();
    const renderCategory = (category) => {
        tabs.replaceChildren();
        for (const name of Object.keys(TAG_LIBRARY)) {
            const tab = document.createElement("button");
            tab.type = "button";
            tab.textContent = name;
            tab.style.cssText = `border:1px solid #666;border-radius:6px;padding:5px 7px;cursor:pointer;background:${name===category?"#526b91":"#333"};color:#fff`;
            tab.addEventListener("pointerdown", (e) => e.preventDefault());
            tab.addEventListener("click", () => renderCategory(name));
            tabs.appendChild(tab);
        }
        list.replaceChildren();
        for (const tag of TAG_LIBRARY[category]) {
            const item = document.createElement("button");
            item.type = "button";
            item.textContent = tag;
            item.title = "Insert at the saved cursor position";
            item.style.cssText = "border:1px solid #666;border-radius:6px;padding:8px 5px;cursor:pointer;background:#303030;color:#eee;text-align:left";
            item.addEventListener("pointerdown", (e) => e.preventDefault());
            item.addEventListener("click", () => {
                insertTagAtCursor(node, widgetName, tag, saved);
                close();
            });
            list.appendChild(item);
        }
    };
    renderCategory(Object.keys(TAG_LIBRARY)[0]);

    const custom = document.createElement("div");
    custom.style.cssText = "display:flex;gap:5px;margin-top:10px";
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Custom tag, e.g. [pause=2.5s]";
    input.style.cssText = "min-width:0;flex:1;padding:7px;border-radius:6px;border:1px solid #666;background:#181818;color:#eee";
    const add = document.createElement("button");
    add.type = "button";
    add.textContent = "Insert";
    add.style.cssText = "padding:6px 9px;border-radius:6px;border:1px solid #777;cursor:pointer";
    add.addEventListener("pointerdown", (e) => e.preventDefault());
    add.addEventListener("click", () => {
        let value = input.value.trim();
        if (!value) return;
        if (!value.startsWith("[")) value = "[" + value;
        if (!value.endsWith("]")) value += "]";
        insertTagAtCursor(node, widgetName, value, saved);
        close();
    });
    custom.append(input, add);
    popup.appendChild(custom);

    const footer = document.createElement("div");
    footer.textContent = "Tags are prompt instructions; acoustic realization depends on the model.";
    footer.style.cssText = "margin-top:9px;color:#aaa;font-size:11px;line-height:1.35";
    popup.appendChild(footer);

    document.body.appendChild(popup);
    const rect = buttonWidget?.element?.getBoundingClientRect?.();
    const x = rect ? Math.min(rect.left, window.innerWidth - 305) : Math.max(12, window.innerWidth - 310);
    const y = rect ? Math.min(rect.bottom + 6, window.innerHeight - 430) : 100;
    popup.style.left = `${Math.max(8, x)}px`;
    popup.style.top = `${Math.max(8, y)}px`;

    const outside = (event) => {
        if (!popup.contains(event.target) && event.target !== buttonWidget?.element) {
            close();
            document.removeEventListener("pointerdown", outside, true);
        }
    };
    setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
}

function applyVisibility(node) {
    if (node.comfyClass !== "MiniMaxH3AudioGenerator") return;
    const mode = node.widgets?.find((w) => w.name === "generation_mode")?.value || "Audio Generation";
    const expressive = mode === "Expressive TTS";
    const submode = node.widgets?.find((w) => w.name === "expressive_tts_mode")?.value || "Single Voice";
    const dual = expressive && submode === "Dual Voice Dialogue";
    const single = expressive && !dual;
    const refs = !!node.widgets?.find((w) => w.name === "use_references")?.value;
    setWidgetVisible(node, "prompt", !expressive);
    // `length` is a manual duration control for Audio Generation only.
    // Expressive TTS calculates its frame budget internally via Smart Duration Engine.
    setWidgetVisible(node, "length", !expressive);
    setWidgetVisible(node, "speech_text", single);
    setWidgetVisible(node, "mm_tag_picker_speech", single);
    setWidgetVisible(node, "mm_tag_picker_dialogue", dual);
    setWidgetVisible(node, "expressive_tts_mode", expressive);
    setWidgetVisible(node, "speech_language", expressive);
    setWidgetVisible(node, "custom_speech_language", expressive &&
        (node.widgets?.find((w) => w.name === "speech_language")?.value === "Custom"));
    setWidgetVisible(node, "dialogue_script", dual);
    setWidgetVisible(node, "speaking_speed_hint", dual);
    setWidgetVisible(node, "duration_profile", dual);
    for (const name of ["voice_type", "custom_voice_description", "voice_timbre",
        "expressive_default_style", "expressive_default_emotion", "expressive_default_intensity",
        "expressive_default_speed", "expressive_default_pitch"]) setWidgetVisible(node, name, single);
    for (const name of ["speaking_style", "emotion", "emotion_intensity", "speech_speed",
        "speech_pitch", "voice_reference_mode", "cloning_instructions"]) setWidgetVisible(node, name, false);
    setWidgetVisible(node, "use_references", !expressive);
    setWidgetVisible(node, "ref_image_size", !expressive && refs);
    // Input sockets are real LiteGraph slots, not widgets: the legacy `hidden`
    // flag does not hide them. Remove inactive slots and restore them on mode
    // changes, preserving any existing upstream connection for later restore.
    const visibleInputs = new Set();
    if (dual) {
        visibleInputs.add("ref_audio_1");
        visibleInputs.add("ref_audio_2");
    } else if (single) {
        visibleInputs.add("ref_audio");
    } else if (refs) {
        visibleInputs.add("ref_image");
        visibleInputs.add("ref_audio");
    }
    setInputSet(node, visibleInputs);
    resizeNode(node);
}

function wrapWidgetCallback(node, name) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (!widget || widget._mmVoiceStudioWrapped) return;
    const previous = widget.callback;
    widget.callback = function (...args) {
        const result = typeof previous === "function" ? previous.apply(this, args) : undefined;
        applyVisibility(node);
        return result;
    };
    widget._mmVoiceStudioWrapped = true;
}


function ensureTagPickerButtons(node) {
    if (!node.widgets) return;
    if (!node.widgets.some((w) => w.name === "mm_tag_picker_speech")) {
        node.addWidget("button", "＋ Tags — Speech", null,
            () => openTagPicker(node, "speech_text", node.widgets.find((w) => w.name === "mm_tag_picker_speech")),
            { serialize: false });
        const w = node.widgets[node.widgets.length - 1];
        w.name = "mm_tag_picker_speech";
        w.serialize = false;
    }
    if (!node.widgets.some((w) => w.name === "mm_tag_picker_dialogue")) {
        node.addWidget("button", "＋ Tags — Dialogue", null,
            () => openTagPicker(node, "dialogue_script", node.widgets.find((w) => w.name === "mm_tag_picker_dialogue")),
            { serialize: false });
        const w = node.widgets[node.widgets.length - 1];
        w.name = "mm_tag_picker_dialogue";
        w.serialize = false;
    }
}

app.registerExtension({
    name: "MiniMaxH3.VoiceStudio.ModeVisibility",
    async nodeCreated(node) {
        if (node.comfyClass !== "MiniMaxH3AudioGenerator") return;
        ensureTagPickerButtons(node);
        requestAnimationFrame(() => applyVisibility(node));
        for (const name of ["generation_mode", "expressive_tts_mode", "speech_language", "use_references"]) wrapWidgetCallback(node, name);
    },
    async loadedGraphNode(node) {
        if (node.comfyClass !== "MiniMaxH3AudioGenerator") return;
        applyVisibility(node);
        requestAnimationFrame(() => applyVisibility(node));
    },
});
