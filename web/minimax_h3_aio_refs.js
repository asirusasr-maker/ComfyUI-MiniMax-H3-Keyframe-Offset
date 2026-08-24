import { app } from "../../scripts/app.js";

const REF_SLOTS = [
    { name: "ref_image", type: "IMAGE" },
    { name: "ref_audio", type: "AUDIO" },
];

function findInputIndex(node, name) {
    return (node.inputs || []).findIndex((i) => i.name === name);
}

function insertAfter(node, afterName, name, type) {
    // add at end first
    node.addInput(name, type);
    const from = node.inputs.length - 1;
    const after = findInputIndex(node, afterName);
    if (after < 0 || from <= after + 1) return;

    // move new input to right after `afterName`
    const [inp] = node.inputs.splice(from, 1);
    node.inputs.splice(after + 1, 0, inp);
}

function showRefSlots(node) {
    for (const { name, type } of REF_SLOTS) {
        const idx = findInputIndex(node, name);
        if (idx >= 0) {
            // already there (or ghost) — restore
            const inp = node.inputs[idx];
            inp.type = type;
            inp.hidden = false;
            delete inp._mm_saved_type;
        } else {
            insertAfter(node, "audio_vae", name, type);
        }
    }
}

function hideRefSlots(node) {
    // remove from the end so indices stay valid
    for (let i = REF_SLOTS.length - 1; i >= 0; i--) {
        const name = REF_SLOTS[i].name;
        const idx = findInputIndex(node, name);
        if (idx < 0) continue;
        if (node.inputs[idx].link != null) {
            node.disconnectInput(idx);
        }
        node.removeInput(idx);
    }
}

function setWidgetVisible(node, name, visible) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) return;
    w.hidden = !visible;
    if (w.element) w.element.style.display = visible ? "" : "none";
}

function applyRefsVisibility(node) {
    if (node.comfyClass !== "MiniMaxH3AudioGenerator") return;

    const toggle = node.widgets?.find((w) => w.name === "use_references");
    const on = !!(toggle && toggle.value);

    if (on) showRefSlots(node);
    else hideRefSlots(node);

    setWidgetVisible(node, "ref_image_size", on);

    // keep WIDTH, only refresh HEIGHT
    try {
        const size = node.computeSize();
        if (node.size) {
            node.size[1] = size[1];
        } else {
            node.setSize([size[0], size[1]]);
        }
    } catch (_) {}

    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "MiniMaxH3.AudioGenerator.RefsToggle",

    async nodeCreated(node) {
        if (node.comfyClass !== "MiniMaxH3AudioGenerator") return;

        // ComfyUI creates all INPUT_TYPES slots first — strip refs if toggle is off
        requestAnimationFrame(() => applyRefsVisibility(node));

        const toggle = node.widgets?.find((w) => w.name === "use_references");
        if (!toggle) return;

        const prev = toggle.callback;
        toggle.callback = function (...args) {
            const r = typeof prev === "function" ? prev.apply(this, args) : undefined;
            applyRefsVisibility(node);
            return r;
        };
    },

    async loadedGraphNode(node) {
        if (node.comfyClass !== "MiniMaxH3AudioGenerator") return;
        // After workflow load: restore slots BEFORE links if toggle was on
        applyRefsVisibility(node);
        requestAnimationFrame(() => applyRefsVisibility(node));
    },
});